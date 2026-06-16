"""SimLingo driving model with the public BLUE gate.

The gate evaluates the last language-token hidden state before language
generation. A decision of 1 runs the language-generation path; a decision of 0
runs the direct driving-action path.
"""

import datetime
import json
import os
import random
import time
from pathlib import Path
from pprint import PrettyPrinter
from typing import Dict, Optional, Tuple, List

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from hydra.utils import get_original_cwd

from simlingo_training.models.adaptors.adaptors import DrivingAdaptor, LanguageAdaptor, WaypointInputAdaptor, AdaptorList
from simlingo_training.models.utils import summarise_losses
from simlingo_training.utils.custom_types import (DrivingExample, DrivingInput,
                                                DrivingLabel, DrivingOutput,
                                                TrainingOutput)

from simlingo_training.models.gate import BaseGate, create_gate

pprint = PrettyPrinter().pprint

def decode_uint8(encoded: torch.Tensor) -> List[str]:
    return [row.tobytes().decode("utf-8").rstrip("\0") for row in encoded.cpu().numpy()]

class NormZeroOne(nn.Module):
    def __init__(self, min_max: Tuple[float, float]):
        super().__init__()
        self.register_buffer("min_max", torch.tensor(min_max, dtype=torch.float), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Normalise tensor to [0, 1] using values from min_max"""
        return (x - self.min_max[0]) / (self.min_max[1] - self.min_max[0])


class DrivingModelGate(pl.LightningModule):
    """SimLingo model wrapper that applies the trained BLUE gate per frame."""
    def __init__(
        self,
        cfg_data_module,
        processor,
        cache_dir,
        gate_mode: str = "trained_gate",
        gate_ckpt: Optional[str] = None,
        gate_kwargs: Optional[Dict] = None,
        **cfg,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        for key, value in cfg.items():
            setattr(self, key, value)
            
        self.processor = processor
        
        self.prediction = {}
        
        self.gate_mode = gate_mode
        self.gate_ckpt = gate_ckpt
        
        self.cfg_data_module = cfg_data_module
        
        # Vision encoder.
        self.vision_model = hydra.utils.instantiate(
            self.vision_model,
            cfg_data_module=cfg_data_module,
            processor=self.processor,
            cache_dir=cache_dir,
            _recursive_=False
        )
            
        # Language model.
        self.language_model = hydra.utils.instantiate(
            self.language_model,
            cache_dir=cache_dir,
            _recursive_=False
        )

        self.all_predictions = {}
        self.all_losses = {}
        
        # Driving adaptor.
        driving = DrivingAdaptor(
            self.language_model.hidden_size, 
            speed_wps_mode=self.speed_wps_mode,
            predict_route_as_wps=self.predict_route_as_wps,
        )

        self.adaptors = AdaptorList(
            language=LanguageAdaptor(self.language_model),
            driving=driving,
        )

        # Waypoint input encoder.
        self.wp_encoder = WaypointInputAdaptor(
            token_size=self.language_model.hidden_size,
            hidden_size=256,
            hidden_size2=512,
        )

        if 'tokenizer' in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor

        _gate_kwargs = gate_kwargs if gate_kwargs is not None else {}
        self.gate = create_gate(
            mode=gate_mode,
            hidden_size=self.language_model.hidden_size,
            ckpt_path=gate_ckpt,
            **_gate_kwargs,
        )
        self.gate.eval()
        for param in self.gate.parameters():
            param.requires_grad = False
        
        print(f"[DrivingModelGate] gate_mode={gate_mode}, "
              f"hidden_size: {self.language_model.hidden_size}")


    def forward(self,
        example: DrivingExample,
        return_language: Optional[bool] = None,
        prompt_ids: Optional[Tensor] = None,
    ) -> DrivingOutput:
        """Run the BLUE gated inference path."""
        self.gate_decisions = []
        self.gate_scores = []
        self.gate_inference_time_ms = []
        
        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example
        
        if driving_input is not None:
            adaptor_dict = self.adaptors(example, inference=True)
            adaptor_dict = self.vision_model.image_encoder.replace_placeholder_tokens(
                    adaptor_dict = adaptor_dict,
                    pixel_values = driving_input.camera_images,
                    placeholder_values = driving_input.prompt_inference.placeholder_values,
                    wp_encoder = self.wp_encoder,
                )
            
            input_embeds_all = adaptor_dict["language_inputs"]
            attention_masks = adaptor_dict['language_inputs_mask']

        batch_size = input_embeds_all.size(0)
        decisions = []
        scores = []
        gate_times_ms = []
        for b_idx in range(batch_size):
            input_embed = input_embeds_all[b_idx].unsqueeze(0)
            attention_mask = attention_masks[b_idx].unsqueeze(0)
            with torch.no_grad():
                features_for_gate, _ = self.language_model.forward(
                    embeddings=input_embed,
                    attention_mask=attention_mask,
                )
            last_hidden_state = features_for_gate[:, -1, :]  # [1, hidden_size]
            if last_hidden_state.is_cuda:
                torch.cuda.synchronize(last_hidden_state.device)
            gate_start = time.perf_counter()
            with torch.no_grad():
                decision, prob = self.gate.forward_with_prob(last_hidden_state)
            if last_hidden_state.is_cuda:
                torch.cuda.synchronize(last_hidden_state.device)
            gate_elapsed_ms = (time.perf_counter() - gate_start) * 1000.0
            decisions.append(int(decision.item()))
            scores.append(float(prob.item()))
            gate_times_ms.append(float(gate_elapsed_ms))

        self.gate_decisions = decisions
        self.gate_scores = scores
        self.gate_inference_time_ms = gate_times_ms
        num_cot = sum(decisions)
        num_quiet = batch_size - num_cot
        
        if num_quiet == batch_size:
            self.speed_wps, self.route, self.language = None, None, None
            
            adaptor_features, adaptor_logits = self.forward_model(driving_input, adaptor_dict)
            outputs_by_adaptor = self.adaptors.split_outputs_by_adaptor(adaptor_dict, adaptor_features)
            predictions = self.adaptors.driving.get_predictions(outputs_by_adaptor['driving'])

            for k, v in predictions.items():
                if v is not None:
                    setattr(self, k, v)
            
        elif num_cot == batch_size:
            self.speed_wps, self.route, self.language = None, None, []
            
            for b_idx, (input_embed, attention_mask) in enumerate(zip(input_embeds_all, attention_masks)):
                input_embed = input_embed.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
                if self.language_model.variant == 'OpenGVLab/InternVL2-4B':
                    eos = self.tokenizer.added_tokens_encoder['<|end|>']
                elif self.language_model.variant == 'OpenGVLab/InternVL2-2B':
                    eos = self.tokenizer.added_tokens_encoder['<|im_end|>']
                else:
                    eos = self.tokenizer.eos_token_id

                sampled_tokens, input_embeds = self.language_model.greedy_sample(
                    input_embed,
                    eos_token_id=eos,
                    max_new_tokens=100,
                    input_embed_matrix=self.adaptors.language.embed_tokens.weight,
                    logit_matrix=self.adaptors.language.lm_head.weight,
                    attention_mask=attention_mask,
                )
                
                inputs_driving = self.adaptors.driving(driving_input)
                input_embed_concat = torch.cat((input_embeds, inputs_driving["inputs"][b_idx].unsqueeze(0)), dim=1)
                features, logits = self.language_model.forward(input_embed_concat)

                len_driving = inputs_driving["inputs"].size(1)

                driving_features = features[:, -len_driving:]
                driving_logits = logits[:, -len_driving:]
                predictions = self.adaptors.driving.get_predictions(driving_features, driving_logits)
                    
                for k, v in predictions.items():
                    if v is not None:
                        if hasattr(self, k) and getattr(self, k) is not None:
                            if isinstance(v, torch.Tensor):
                                setattr(self, k, torch.cat((getattr(self, k), v), dim=0))
                            elif isinstance(v, list):
                                getattr(self, k).append(v)
                            else:
                                raise NotImplementedError(f"Type of {k} not supported")
                        else:
                            setattr(self, k, v)
                                
                self.language.append(self.tokenizer.batch_decode(sampled_tokens, skip_special_tokens=True)[0])
        
        else:
            # Pass 1 runs the direct-action path for all samples.
            # Pass 2 reruns the language-generation path only for samples
            # selected by the gate and overwrites their direct-action output.
            self.speed_wps, self.route, self.language = None, None, []
            
            # Pass 1: direct-action forward for all samples.
            adaptor_features, adaptor_logits = self.forward_model(driving_input, adaptor_dict)
            outputs_by_adaptor = self.adaptors.split_outputs_by_adaptor(adaptor_dict, adaptor_features)
            quiet_predictions = self.adaptors.driving.get_predictions(outputs_by_adaptor['driving'])
            
            for k, v in quiet_predictions.items():
                if v is not None:
                    setattr(self, k, v)
            
            # --- Pass 2: CoT override for selected samples ---
            for b_idx in range(batch_size):
                if decisions[b_idx] != 1:
                    self.language.append(None)
                    continue
                
                # Language-generation path for the selected sample.
                input_embed = input_embeds_all[b_idx].unsqueeze(0)
                attention_mask = attention_masks[b_idx].unsqueeze(0)
                
                if self.language_model.variant == 'OpenGVLab/InternVL2-4B':
                    eos = self.tokenizer.added_tokens_encoder['<|end|>']
                elif self.language_model.variant == 'OpenGVLab/InternVL2-2B':
                    eos = self.tokenizer.added_tokens_encoder['<|im_end|>']
                else:
                    eos = self.tokenizer.eos_token_id

                sampled_tokens, input_embeds = self.language_model.greedy_sample(
                    input_embed,
                    eos_token_id=eos,
                    max_new_tokens=100,
                    input_embed_matrix=self.adaptors.language.embed_tokens.weight,
                    logit_matrix=self.adaptors.language.lm_head.weight,
                    attention_mask=attention_mask,
                )
                
                inputs_driving = self.adaptors.driving(driving_input)
                input_embed_concat = torch.cat((input_embeds, inputs_driving["inputs"][b_idx].unsqueeze(0)), dim=1)
                features, logits = self.language_model.forward(input_embed_concat)

                len_driving = inputs_driving["inputs"].size(1)
                driving_features = features[:, -len_driving:]
                driving_logits = logits[:, -len_driving:]
                cot_predictions = self.adaptors.driving.get_predictions(driving_features, driving_logits)
                
                # Overwrite the direct-action prediction for this sample.
                for k, v in cot_predictions.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        current = getattr(self, k, None)
                        if current is not None:
                            current[b_idx] = v[0]
                
                commentary = self.tokenizer.batch_decode(sampled_tokens, skip_special_tokens=True)[0]
                self.language.append(commentary)

                print(f"[BlueGate] batch decisions: generate={num_cot}, direct={num_quiet}, total={batch_size}")

        return self.speed_wps, self.route, self.language


    def forward_model(self, 
                      driving_input: DrivingInput, 
                      adaptor_dict: Dict, 
                      driving_labels: DrivingLabel = None,
                      ) -> Tensor:
        """Forward model conditioned on the given driving input."""
        
        adaptor_dict = self.vision_model.image_encoder.replace_placeholder_tokens(
            adaptor_dict = adaptor_dict,
            pixel_values = driving_input.camera_images,
            placeholder_values = driving_input.prompt.placeholder_values,
            wp_encoder = self.wp_encoder,
        )

        position_ids = None
        adaptor_embeds = adaptor_dict["inputs"]
        adaptor_mask = adaptor_dict['inputs_mask']

        input_embeds = adaptor_embeds
        input_embeds = input_embeds.to(
            dtype=self.language_model.model.dtype
        )
        attention_mask = adaptor_mask

        outputs = self.language_model.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            output_hidden_states=True,
            return_dict=True,
        )
        features = outputs.hidden_states[-1]
        logits = outputs[0]

        vision_features, adaptor_features = features.split(
            [features.size(1) - adaptor_embeds.size(1), adaptor_embeds.size(1)], dim=1
        )
        vision_logits, adaptor_logits = logits.split(
            [logits.size(1) - adaptor_embeds.size(1), adaptor_embeds.size(1)], dim=1
        )
        return adaptor_features, adaptor_logits
    

    def forward_loss(self, example: DrivingExample, per_sample=False) -> TrainingOutput:
        """
        Forward pass of the model for a driving input, followed by
        computing the next token cross-entropy loss.
        """

        adaptor_dict = self.adaptors(example)
        adaptor_embeds = adaptor_dict["inputs"]
        adaptor_mask = adaptor_dict['inputs_mask']

        adaptor_features, adaptor_logits = self.forward_model(example.driving_input, adaptor_dict, driving_labels=example.driving_label)
        loss_dict = self.adaptors.compute_loss(adaptor_features, adaptor_logits, adaptor_dict, example)

        loss_dict_only_losses = {k:v for k, v in loss_dict.items() if k.endswith("loss")}
        loss_logs = {k:v for k, v in loss_dict.items() if k.endswith("log")}
        
        pred_labels = {k:v for k, v in loss_dict.items() if not k.endswith("loss") and not k.endswith("log")}
        if per_sample:
            return loss_dict_only_losses, pred_labels

        return summarise_losses(loss_dict_only_losses), loss_logs

    def training_step(self, batch: DrivingExample, _batch_idx: int = 0):
        output, loss_logs = self.forward_loss(batch)
        logs = output
        self.log_training_output(logs, "train")
        self.log("train/loss", output.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": output.loss, "outputs": output}


    def validation_step(self, batch: DrivingExample, _batch_idx: int = 0):
        output, loss_logs = self.forward_loss(batch)
        logs = output
        self.log_training_output(logs, "val")
        self.log("val/loss", output.loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": output.loss, "outputs": output}

    def predict_step(self, batch: DrivingExample, _batch_idx: int = 0):
        """Run prediction and store BLUE gate decisions."""
        run_ids = decode_uint8(batch.run_id)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t_start = time.perf_counter()
        speed_wps, route, language = self.forward(batch, return_language=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t_end = time.perf_counter()
        _batch_time = _t_end - _t_start
        _batch_frames = len(run_ids)
        if not hasattr(self, '_timing_total_time'):
            self._timing_total_time = 0.0
            self._timing_total_frames = 0
        self._timing_total_time += _batch_time
        self._timing_total_frames += _batch_frames

        self.num_route_points = 20
        route_equal = []
        for i in range(len(route)):
            route_equal.append(self.equal_spacing_route(route[i].cpu()))
        route_equal = torch.tensor(route_equal)
        route = route_equal.to(route.device)
        
        route_gt = batch.driving_label.path
        speed_wps_gt = batch.driving_label.waypoints
        language_gt = batch.driving_label.answer.language_string
        
        # Keep language predictions as a list. Direct-action samples use an
        # empty string because they intentionally skip language generation.
        if language is not None:
            language_for_pred = [l if l is not None else "" for l in language]
        else:
            language_for_pred = [""] * len(language_gt)
        
        if len(self.prediction) == 0:
            self.prediction = {
                "waypoints": [speed_wps],
                "route": [route],
                "language": language_for_pred,
                "waypoints_gt": [speed_wps_gt],
                "route_gt": [route_gt],
                "language_gt": language_gt,
                "prompt": batch.driving_input.prompt.language_string,
                "path": run_ids,
                "qa_templates": batch.qa_templates,
                "eval_infos": batch.driving_label.eval_infos,
                "gate_decisions": list(self.gate_decisions),
                "gate_scores": list(self.gate_scores),
                "gate_inference_time_ms": list(self.gate_inference_time_ms),
            }
        else:
            self.prediction["waypoints"].append(speed_wps)
            self.prediction["route"].append(route)
            if language_for_pred is not None:
                self.prediction["language"].extend(language_for_pred)
            self.prediction["waypoints_gt"].append(speed_wps_gt)
            self.prediction["route_gt"].append(route_gt)
            self.prediction["language_gt"].extend(language_gt)
            self.prediction["prompt"].extend(batch.driving_input.prompt.language_string)
            self.prediction["path"].extend(run_ids)
            self.prediction["qa_templates"].extend(batch.qa_templates)
            self.prediction["eval_infos"].extend(batch.driving_label.eval_infos)
            self.prediction["gate_decisions"].extend(list(self.gate_decisions))
            self.prediction["gate_scores"].extend(list(self.gate_scores))
            self.prediction["gate_inference_time_ms"].extend(list(self.gate_inference_time_ms))
            
        return speed_wps, route, language, speed_wps_gt, route_gt, language_gt

    def equal_spacing_route(self, points):
        route = np.concatenate((np.zeros_like(points[:1]),  points))
        shift = np.roll(route, 1, axis=0)
        shift[0] = shift[1]

        dists = np.linalg.norm(route-shift, axis=1)
        dists = np.cumsum(dists)
        dists += np.arange(0, len(dists))*1e-4

        x = np.arange(0, 20, 1)
        interp_points = np.array([np.interp(x, dists, route[:, 0]), np.interp(x, dists, route[:, 1])]).T

        return interp_points

    def on_predict_epoch_end(self) -> None:
        """Save prediction outputs and BLUE gate statistics."""
        repo_path = get_original_cwd()

        if self.trainer.ckpt_path is not None:
            ckpt_path = Path(self.trainer.ckpt_path).parent.parent
        else:
            ckpt_path = Path(f'{repo_path}/outputs/{self.language_model.variant}')
        save_prediction_path = ckpt_path / "predictions"
        save_prediction_path.mkdir(exist_ok=True, parents=True)
        
        if hasattr(self, '_timing_total_time') and self._timing_total_frames > 0:
            avg_latency = self._timing_total_time / self._timing_total_frames
            fps = 1.0 / avg_latency if avg_latency > 0 else float('inf')
            timing_stats = {
                "total_inference_time_s": self._timing_total_time,
                "total_frames": self._timing_total_frames,
                "avg_latency_per_frame_ms": avg_latency * 1000,
                "fps_per_frame": fps,
                "gate_mode": self.gate_mode,
            }
            print(f"\n{'='*60}")
            print(f"  Inference Timing (per-frame, ignoring batch parallelism)")
            print(f"{'='*60}")
            print(f"  Total time:           {self._timing_total_time:.2f} s")
            print(f"  Total frames:         {self._timing_total_frames}")
            print(f"  Avg latency/frame:    {avg_latency*1000:.2f} ms")
            print(f"  FPS (per-frame):      {fps:.2f}")
            print(f"  Gate mode:            {self.gate_mode}")
            print(f"{'='*60}\n")
            timing_path = save_prediction_path / f"timing_rank_{self.local_rank}.json"
            if timing_path.exists():
                ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                timing_path = save_prediction_path / f"timing_rank_{self.local_rank}_{ts}.json"
            with open(timing_path, "w") as f:
                json.dump(timing_stats, f, indent=4)

        if "gate_decisions" in self.prediction:
            decisions = self.prediction["gate_decisions"]
            num_cot = sum(decisions)
            num_quiet = len(decisions) - num_cot
            gate_stats = {
                "gate_mode": self.gate_mode,
                "total_samples": len(decisions),
                "generate_samples": num_cot,
                "direct_samples": num_quiet,
                "generate_ratio": num_cot / max(len(decisions), 1),
                "decisions": decisions,
                "scores": self.prediction.get("gate_scores", []),
                "inference_time_ms": self.prediction.get("gate_inference_time_ms", []),
            }
            save_path_tmp = f"{str(save_prediction_path)}/blue_gate_decisions_rank_{self.local_rank}.json"
            if os.path.exists(save_path_tmp):
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path_tmp = f"{str(save_prediction_path)}/blue_gate_decisions_rank_{self.local_rank}_{time}.json"
            with open(save_path_tmp, "w") as f:
                json.dump(gate_stats, f, indent=4)
            print(f"[BlueGate] Saved gate statistics: generate={num_cot}, direct={num_quiet}, "
                  f"generate_ratio={num_cot/max(len(decisions),1):.2%}")
        samples_cot = [i for i, l in enumerate(self.prediction["prompt"]) if "What should the ego do next?" in l]
        samples_qa = [i for i, l in enumerate(self.prediction["prompt"]) if "Q:" in l]
        samples_all = [i for i in range(len(self.prediction["prompt"]))]

        # Handle None language (all-Quiet mode produces no language output)
        pred_language = self.prediction["language"]
        if pred_language is None:
            num_samples = len(self.prediction["path"])
            pred_language = [""] * num_samples
            self.prediction["language"] = pred_language

        language = [(l, l_gt, p) for l, l_gt, p in zip(pred_language, self.prediction["language_gt"], self.prediction["path"])]
        
        if len(samples_qa) > 0:
            sorted_samples = {}
            for qa_template, language_sample in zip(self.prediction["qa_templates"], language):
                question = qa_template[0]
                answer = qa_template[1]
                if question not in sorted_samples:
                    sorted_samples[question] = {}
                if answer not in sorted_samples[question]:
                    sorted_samples[question][answer] = []
                sorted_samples[question][answer].append(language_sample)
            
            if os.path.exists(f"{str(save_prediction_path)}/sorted_qa_templates_rank_{self.local_rank}.json"):
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                with open(f"{str(save_prediction_path)}/sorted_qa_templates_rank_{self.local_rank}_{time}.json", "w") as f:
                    json.dump(sorted_samples, f, indent=4)
            else:
                with open(f"{str(save_prediction_path)}/sorted_qa_templates_rank_{self.local_rank}.json", "w") as f:
                    json.dump(sorted_samples, f, indent=4)
        
        for samples, name in zip([samples_cot, samples_qa, samples_all], ["cot", "qa", "all"]):
            language_samples = [l for i, l in enumerate(language) if i in samples]
        
            save_path_tmp = f"{str(save_prediction_path)}/language_preds_{name}_rank_{self.local_rank}.json"
            if os.path.exists(save_path_tmp):
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path_tmp = f"{str(save_prediction_path)}/language_preds_{name}_rank_{self.local_rank}_{time}.json"
            with open(save_path_tmp, "w") as f:
                json.dump(language_samples, f, indent=4)
            
        route_preds = self.prediction["route"]
        route_preds = torch.cat(route_preds, dim=0)
        route_gt = self.prediction["route_gt"]
        route_gt = torch.cat(route_gt, dim=0)
        
        waypoints_preds = self.prediction["waypoints"]
        waypoints_preds = torch.cat(waypoints_preds, dim=0)
        waypoints_gt = self.prediction["waypoints_gt"]
        waypoints_gt = torch.cat(waypoints_gt, dim=0)
        
        waypoints_preds_1d = []
        for i in range(len(waypoints_preds)):
            waypoint_pred = waypoints_preds[i]
            waypoints_preds_1d_tmp = torch.tensor([torch.linalg.norm(waypoint_pred[j+1] - waypoint_pred[j]) for j in range(len(waypoint_pred)-1)])
            waypoints_preds_1d_tmp = torch.cumsum(waypoints_preds_1d_tmp, dim=0)
            waypoints_preds_1d_tmp = [[0, 0]] + [[x, 0] for x in waypoints_preds_1d_tmp]
            waypoints_preds_1d.append(waypoints_preds_1d_tmp)
        waypoints_preds_1d = torch.tensor(waypoints_preds_1d)
        
        waypoints_gt_1d = []
        for i in range(len(waypoints_gt)):
            waypoint_gt = waypoints_gt[i]
            waypoints_gt_1d_tmp = torch.tensor([torch.linalg.norm(waypoint_gt[j+1] - waypoint_gt[j]) for j in range(len(waypoint_gt)-1)])
            waypoints_gt_1d_tmp = torch.cumsum(waypoints_gt_1d_tmp, dim=0)
            waypoints_gt_1d_tmp = [[0, 0]] + [[x, 0] for x in waypoints_gt_1d_tmp]
            waypoints_gt_1d.append(waypoints_gt_1d_tmp)
        waypoints_gt_1d = torch.tensor(waypoints_gt_1d)
        
        samples_safety = [i for i, l in enumerate(self.prediction["prompt"]) if "<SAFETY>" in l]
        samples_instruction = [i for i, l in enumerate(self.prediction["prompt"]) if "<INSTRUCTION_FOLLOWING>" in l]
        samples_neither = [i for i, l in enumerate(self.prediction["prompt"]) if "<SAFETY>" not in l and "<INSTRUCTION_FOLLOWING>" not in l]
        samples_all = [i for i in range(len(self.prediction["prompt"]))]
        
        ade_fde = {}

        # ---- Open-Loop L2 metrics on waypoints (all samples) ----
        # Our waypoints are at 4 Hz (every 0.25s, wp_subsample=5, CARLA 20fps).
        # Paper convention (UniAD / Bench2Drive): report at **2 Hz** over 2s
        #   → 4 evaluation points at 0.5s, 1.0s, 1.5s, 2.0s
        #   → at 4Hz these are indices 1, 3, 5, 7
        #   → "Avg. L2" = mean L2 of these 4 points
        wp_pred_np = waypoints_preds.cpu().numpy()
        wp_gt_np = waypoints_gt.cpu().numpy()
        wp_l2 = np.linalg.norm(wp_pred_np - wp_gt_np, axis=-1)  # [N, pred_len]
        wp_l2_per_step = np.mean(wp_l2, axis=0)  # [pred_len]
        
        wp_hz = 4  # native waypoint frequency (4 Hz)
        ade_fde["wp_pred_len"] = int(wp_l2.shape[1])
        
        # --- Paper-aligned metrics (2 Hz, following UniAD/Bench2Drive) ---
        # Evaluation timestamps at 2Hz: 0.5s, 1.0s, 1.5s, 2.0s
        # At 4Hz native rate: indices 1, 3, 5, 7
        eval_2hz_indices = []
        eval_2hz_times = [0.5, 1.0, 1.5, 2.0]
        for t in eval_2hz_times:
            idx = int(t * wp_hz) - 1  # 0.5s->1, 1.0s->3, 1.5s->5, 2.0s->7
            if idx < wp_l2.shape[1]:
                eval_2hz_indices.append(idx)
                ade_fde[f"L2_{t:.1f}s"] = float(wp_l2_per_step[idx])
        # "Avg. L2" — the headline metric reported in papers
        if len(eval_2hz_indices) > 0:
            avg_l2_2hz = float(np.mean(wp_l2_per_step[eval_2hz_indices]))
            ade_fde["Avg_L2"] = avg_l2_2hz
        
        # --- Native 4Hz metrics (for completeness) ---
        ade_fde["wp_avg_l2_4hz"] = float(np.mean(wp_l2))
        for t_sec in [1.0, 2.0]:
            t_idx = int(t_sec * wp_hz) - 1  # 1s->idx3, 2s->idx7
            if t_idx < wp_l2.shape[1]:
                ade_fde[f"wp_l2_{t_sec:.0f}s"] = float(wp_l2_per_step[t_idx])
                ade_fde[f"wp_ade_{t_sec:.0f}s"] = float(np.mean(wp_l2[:, :t_idx + 1]))
                ade_fde[f"wp_fde_{t_sec:.0f}s"] = float(np.mean(wp_l2[:, t_idx]))
        
        # Route metrics
        route_l2 = np.linalg.norm(route_preds.cpu().numpy() - route_gt.cpu().numpy(), axis=-1)
        ade_fde["route_ade"] = float(np.mean(route_l2))
        ade_fde["route_fde"] = float(np.mean(route_l2[:, -1]))
        ade_fde["gate_mode"] = self.gate_mode
        ade_fde["num_samples_total"] = int(wp_l2.shape[0])
        
        # --- Per-split L2 (SAFETY vs INSTRUCTION) ---
        # SAFETY samples use original GT trajectory; INSTRUCTION samples use
        # modified GT trajectory (lane change, stop, speed change, etc.).
        # Per-split L2 helps diagnose: SAFETY L2 ≈ paper baselines,
        # while INSTRUCTION L2 is higher because model may not follow instructions.
        samples_safety = [i for i, l in enumerate(self.prediction["prompt"]) if "<SAFETY>" in l]
        samples_instruction = [i for i, l in enumerate(self.prediction["prompt"]) if "<INSTRUCTION_FOLLOWING>" in l]
        for split_name, split_indices in [("safety", samples_safety), ("instruction", samples_instruction)]:
            if len(split_indices) == 0:
                continue
            split_l2 = wp_l2[split_indices]  # [n_split, pred_len]
            split_l2_per_step = np.mean(split_l2, axis=0)
            # Paper-aligned 2Hz Avg. L2 for this split
            valid_2hz = [idx for idx in eval_2hz_indices if idx < split_l2.shape[1]]
            if len(valid_2hz) > 0:
                ade_fde[f"Avg_L2_{split_name}"] = float(np.mean(split_l2_per_step[valid_2hz]))
            for t in eval_2hz_times:
                idx = int(t * wp_hz) - 1
                if idx < split_l2.shape[1]:
                    ade_fde[f"L2_{t:.1f}s_{split_name}"] = float(split_l2_per_step[idx])
        
        # Print paper-aligned metrics prominently
        print(f"\n{'='*60}")
        print(f"  Open-Loop Results (paper-aligned, 2Hz over 2s)")
        print(f"{'='*60}")
        print(f"  Avg. L2 (all):     {ade_fde.get('Avg_L2', 'N/A'):.4f} m")
        for t in eval_2hz_times:
            key = f"L2_{t:.1f}s"
            print(f"  L2@{t:.1f}s:         {ade_fde.get(key, 'N/A'):.4f} m" if key in ade_fde else f"  L2@{t:.1f}s:         N/A")
        print(f"  Route ADE:         {ade_fde['route_ade']:.4f} m")
        print(f"  Route FDE:         {ade_fde['route_fde']:.4f} m")
        print(f"  N={wp_l2.shape[0]}, gate={self.gate_mode}")
        # Per-split breakdown
        for sn in ["safety", "instruction"]:
            avg_key = f"Avg_L2_{sn}"
            if avg_key in ade_fde:
                print(f"  --- {sn.upper()} split ---")
                print(f"  Avg. L2 ({sn}): {ade_fde[avg_key]:.4f} m")
                for t in eval_2hz_times:
                    k = f"L2_{t:.1f}s_{sn}"
                    if k in ade_fde:
                        print(f"  L2@{t:.1f}s ({sn}): {ade_fde[k]:.4f} m")
        print(f"{'='*60}\n")

        def get_desired_end_speed(wps):
            wp_freq = 5
            carla_fps = 20
            last_wp = wps[-1]
            one_second = int(carla_fps // (wp_freq))
            half_second = one_second // 2
            prev_wp = wps[-1 - half_second]
            desired_speed = np.linalg.norm(prev_wp - last_wp) * 2.0
            return desired_speed
        def get_desired_speed(wps):
            wp_freq = 5
            carla_fps = 20
            one_second = int(carla_fps // (wp_freq))
            half_second = one_second // 2
            wp_half_second = wps[half_second]
            wp_one_second = wps[one_second]
            desired_speed = np.linalg.norm(wp_half_second - wp_one_second) * 2.0
            return desired_speed
        
        def get_desired_avg_speed(wps):
            first_wp = wps[0]
            last_wp = wps[-1]
            desired_speed = np.linalg.norm(first_wp - last_wp) / (len(wps) * 0.25)
            return desired_speed
        
        def get_1d_wps(wps):
            waypoints_1d = [np.linalg.norm(wps[i+1] - wps[i]) for i in range(len(wps)-1)]
            waypoints_1d = np.cumsum(waypoints_1d)
            waypoints_1d = [[x, 0] for x in waypoints_1d]
            waypoints_1d = [[0, 0]] + waypoints_1d
            return np.array(waypoints_1d).reshape(-1, 2)
        
        wp_freq = 5
        carla_fps = 20
            
        num_total = len(self.prediction["prompt"])
        eval_infos_list = self.prediction["eval_infos"] if isinstance(self.prediction["eval_infos"], list) else []
        has_eval_infos = len(eval_infos_list) > 0 and any(e is not None for e in eval_infos_list)
        num_eval_infos = len(eval_infos_list) if has_eval_infos else 0
        print(f"[Debug] num_total={num_total}, num_eval_infos={num_eval_infos}, has_eval_infos={has_eval_infos}, "
              f"waypoints_preds.shape={waypoints_preds.shape}, route_preds.shape={route_preds.shape}")

        if not has_eval_infos:
            # OpenLoop mode: no instruction-following eval_infos, skip success rate loop
            print("[Info] No eval_infos (OpenLoop mode). Skipping instruction-following success rate evaluation.")
        
        for samples, name in zip([samples_safety, samples_instruction, samples_neither, samples_all], ["safety", "instruction", "neither", "all"]):
            if len(samples) == 0:
                continue
            max_idx = max(samples)
            if has_eval_infos and (max_idx >= waypoints_preds.shape[0] or max_idx >= num_eval_infos):
                print(f"[WARNING] Skipping '{name}': max sample idx {max_idx} >= "
                      f"waypoints len {waypoints_preds.shape[0]} or eval_infos len {num_eval_infos}")
                continue
            route_preds_sample = route_preds[samples].cpu().numpy()
            route_gt_sample = route_gt[samples].cpu().numpy()
            waypoints_preds_sample = waypoints_preds[samples].cpu().numpy()
            waypoints_gt_sample = waypoints_gt[samples].cpu().numpy()
            prompts = [self.prediction["prompt"][i].replace("<IMG_CONTEXT>", "") for i in samples]
            pred_language = [self.prediction["language"][i] for i in samples]
            paths = [self.prediction["path"][i] for i in samples]

            if has_eval_infos:
                eval_infos_sample = [self.prediction["eval_infos"][i] for i in samples]
                waypoints_org_sample = [eval_infos_sample[i]["org_wps"] for i in range(len(samples))]
                route_org_sample = [eval_infos_sample[i]["org_path"] for i in range(len(samples))]
                waypoints_instruction_sample = [np.array(eval_infos_sample[i]["new_wps"]) for i in range(len(samples))]
                route_instruction_sample = [eval_infos_sample[i]["new_path"] for i in range(len(samples))]
            else:
                eval_infos_sample = None
            
            success_rate_all = []
            success_rate_by_mode = {}
            success_rate_by_allowed = {}
            
            paths_by_mode = {}
            
            if not has_eval_infos:
                # OpenLoop mode: skip instruction-following success rate per-sample loop
                pass
            else:
              for i in range(len(samples)):
                mode = eval_infos_sample[i]["mode"]
                allowed = eval_infos_sample[i]['allowed']
                sample_path = paths[i]
                
                if mode not in success_rate_by_mode:
                    success_rate_by_mode[mode] = []
                if mode not in paths_by_mode:
                    paths_by_mode[mode] = []
                
                if allowed not in success_rate_by_allowed:
                    success_rate_by_allowed[allowed] = []
                
                desired_end_speed_pred = get_desired_end_speed(waypoints_preds_sample[i])
                desired_end_speed_gt = get_desired_end_speed(waypoints_gt_sample[i])
                desired_end_speed_org = get_desired_end_speed(waypoints_org_sample[i])
                desired_end_speed_instruction = get_desired_end_speed(waypoints_instruction_sample[i])
                
                desired_speed_pred = get_desired_speed(waypoints_preds_sample[i])
                desired_speed_gt = get_desired_speed(waypoints_gt_sample[i])
                desired_speed_org = get_desired_speed(waypoints_org_sample[i])
                desired_speed_instruction = get_desired_speed(waypoints_instruction_sample[i])
                
                desired_avg_speed_pred = get_desired_avg_speed(waypoints_preds_sample[i])
                desired_avg_speed_gt = get_desired_avg_speed(waypoints_gt_sample[i])
                desired_avg_speed_org = get_desired_avg_speed(waypoints_org_sample[i])
                desired_avg_speed_instruction = get_desired_avg_speed(waypoints_instruction_sample[i])

                
                pred_wps_1d = get_1d_wps(waypoints_preds_sample[i])
                pred_wps_1d_diffs = np.diff(pred_wps_1d[:, 0])
                pred_speeds = pred_wps_1d_diffs / (wp_freq/carla_fps)
                
                org_wps_1d = get_1d_wps(waypoints_org_sample[i])
                org_wps_1d_diffs = np.diff(org_wps_1d[:, 0])
                org_speeds = org_wps_1d_diffs / (wp_freq/carla_fps)
                
                instruction_wps_1d = get_1d_wps(waypoints_instruction_sample[i])
                instruction_wps_1d_diffs = np.diff(instruction_wps_1d[:, 0])
                instruction_speeds = instruction_wps_1d_diffs / (wp_freq/carla_fps)
                
                x = np.arange(len(pred_speeds))*0.25
                
                slope_pred, intercept_pred = np.polyfit(x, pred_speeds, 1)
                slope_org, intercept_org = np.polyfit(x, org_speeds, 1)
                slope_instruction, intercept_instruction = np.polyfit(x, instruction_speeds, 1)
                
                current_speed = float(prompts[i].split("Current speed: ")[-1].split(" ")[0])
                
                if mode == 'stop':
                    paths_by_mode[mode].append(sample_path)
                    if name == 'instruction' or name == 'neither':
                        if np.min(pred_speeds) < 0.1:
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                            
                elif mode == 'slower':
                    paths_by_mode[mode].append(sample_path)

                    if name == 'instruction' or name == 'neither':
                        if slope_pred < (-0.05 * current_speed):
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                        
                elif mode == 'faster':
                    paths_by_mode[mode].append(sample_path)
                    
                    if name == 'instruction' or name == 'neither':
                        if slope_pred > (0.05 * current_speed):
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                elif mode == 'target_speed':
                    paths_by_mode[mode].append(sample_path)
                    
                    try:
                        target_speed = float(prompts[i].split("Target waypoint: ")[-1].split("Command")[-1].split(".<|im_end|>")[0].split(" ")[-2])
                    except:
                        target_speed = float(prompts[i].split("Target waypoint: ")[-1].split("Command")[-1].split(".<|im_end|>")[0].split(" ")[-3])
                    if name == 'instruction' or name == 'neither':
                        if ((desired_end_speed_pred > 0.8 * desired_end_speed_instruction and desired_end_speed_pred < 1.2 * desired_end_speed_instruction) or (desired_end_speed_pred > 0.8 * target_speed and desired_end_speed_pred < 1.2 * target_speed)):
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                    
                elif mode == 'lane_change':
                    paths_by_mode[mode].append(sample_path)
                    
                    fde_pred_org = np.linalg.norm(route_preds_sample[i][-1] - route_org_sample[i][-1], axis=-1)
                    fde_pred_instruction = np.linalg.norm(route_preds_sample[i][-1] - route_instruction_sample[i][-1], axis=-1)
                    if name == 'instruction' or name == 'neither':
                        if fde_pred_instruction < fde_pred_org:
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                elif mode == 'crash':
                    paths_by_mode[mode].append(sample_path)
                    
                    ade_path_org_instruction = np.mean(np.linalg.norm(route_org_sample[i] - route_instruction_sample[i], axis=-1))
                    ade_path_pred_org = np.mean(np.linalg.norm(route_preds_sample[i] - route_org_sample[i], axis=-1))
                    ade_path_pred_instruction = np.mean(np.linalg.norm(route_preds_sample[i] - route_instruction_sample[i], axis=-1))
                    if ade_path_org_instruction > 1.0:
                        if name == 'instruction' or name == 'neither':
                            if ade_path_pred_instruction < ade_path_pred_org:
                                success_rate_all.append(1)
                                success_rate_by_mode[mode].append(1)
                                success_rate_by_allowed[allowed].append(1)
                            else:
                                success_rate_all.append(0)
                                success_rate_by_mode[mode].append(0)
                                success_rate_by_allowed[allowed].append(0)
                    else:
                        if name == 'instruction' or name == 'neither':
                            if ade_path_pred_instruction < 1.0 and (np.mean(pred_speeds) < 1.3 * np.mean(instruction_speeds) or np.mean(pred_speeds) > 0.7 * np.mean(instruction_speeds)):
                                success_rate_all.append(1)
                                success_rate_by_mode[mode].append(1)
                                success_rate_by_allowed[allowed].append(1)
                            else:
                                success_rate_all.append(0)
                                success_rate_by_mode[mode].append(0)
                                success_rate_by_allowed[allowed].append(0)

                else:
                    print(f"Unknown mode: {mode} in sample {i} with path {sample_path}")
                                
            per_sample_results = {
                'paths_by_mode': paths_by_mode,
                'success_rate_by_mode': success_rate_by_mode
            }
            if has_eval_infos:
                save_path_tmp = f"{str(save_prediction_path)}/results_per_sample_{name}_rank_{self.local_rank}.json"
                if os.path.exists(save_path_tmp):
                    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    save_path_tmp = f"{str(save_prediction_path)}/results_per_sample_{name}_rank_{self.local_rank}_{time}.json"
                with open(save_path_tmp, "w") as f:
                    json.dump(per_sample_results, f, indent=4)
            
            if len(success_rate_all) > 0:
                total_success_rate = sum(success_rate_all) / len(success_rate_all)
                ade_fde.update({f"success_rate_total_{name}": total_success_rate})
                
            if len(success_rate_by_mode) > 0:
                for mode in success_rate_by_mode:
                    if len(success_rate_by_mode[mode]) > 0:
                        success_rate = sum(success_rate_by_mode[mode]) / len(success_rate_by_mode[mode])
                        ade_fde.update({f"success_rate_{name}_{mode}": success_rate})
                    else:
                        ade_fde.update({f"success_rate_{name}_{mode}": 0})
                
            ade_route = np.mean(np.linalg.norm(route_preds_sample - route_gt_sample, axis=-1), axis=-1)
            
            ade_fde.update({
                f"num_samples_{name}": len(ade_route),
            })

        # Choose output filename based on whether eval_infos exist
        result_basename = "dreamer_results" if has_eval_infos else "openloop_results"
        save_path_tmp = f"{str(save_prediction_path)}/{result_basename}_rank_{self.local_rank}.json"
        if os.path.exists(save_path_tmp):
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path_tmp = f"{str(save_prediction_path)}/{result_basename}_rank_{self.local_rank}_{time}.json"
        with open(save_path_tmp, "w") as f:
            json.dump(ade_fde, f, indent=4)
        

    def log_training_output(self, training_output: TrainingOutput, mode: str, dataset: Optional[str] = None):
        losses = {k: n.detach() for k, n in training_output.loss_averages.items()}
        counts = {k: n.detach().sum() for k, n in training_output.loss_counts.items()}
        losses["loss"] = training_output.loss.detach()
        counts["loss"] = 1
        for k, v in sorted(losses.items()):
            log_key = f"{mode}_losses/{k}"
            self.log(log_key, v, batch_size=counts[k], sync_dist=True, add_dataloader_idx=False)


    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        if self.trainer.max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
        else:
            max_steps = self.trainer.max_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=max_steps, pct_start=self.pct_start, verbose=False
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "step"}}
