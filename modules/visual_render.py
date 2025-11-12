# ============================================================
# Kalki v2.4 — visual_render.py
# ------------------------------------------------------------
# Visual Render Engine: Photorealistic Design Visualization
# - ComfyUI/SDXL integration for AI rendering
# - Material and lighting setup
# - Multi-angle rendering
# - Animation sequence generation
# ============================================================

import os
import json
import asyncio
import base64
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import requests

from modules.utils.config import get_config
from modules.utils.logging_config import get_logger

logger = get_logger("Kalki.VisualRender")

@dataclass
class RenderJob:
    """Rendering job specification"""
    job_id: str
    design_id: str
    render_type: str  # "photorealistic", "technical", "animation"
    parameters: Dict[str, Any]
    status: str  # "pending", "running", "completed", "failed"
    output_files: List[str]

@dataclass
class RenderScene:
    """3D scene setup for rendering"""
    camera_positions: List[Tuple[float, float, float]]
    lighting_setup: Dict[str, Any]
    materials: Dict[str, Dict[str, Any]]
    environment: Dict[str, Any]

class VisualRenderEngine:
    """AI-powered visual rendering engine"""

    def __init__(self):
        self.comfyui_endpoint = "http://localhost:8188"  # Default ComfyUI endpoint
        self.render_jobs = {}
        self.templates = self._load_render_templates()

    def _load_render_templates(self) -> Dict[str, Any]:
        """Load rendering templates and workflows"""

        return {
            "photorealistic": {
                "workflow": {
                    "1": {
                        "class_type": "CheckpointLoaderSimple",
                        "inputs": {
                            "ckpt_name": "sdxl_base_1.0.safetensors"
                        }
                    },
                    "2": {
                        "class_type": "CLIPTextEncode",
                        "inputs": {
                            "text": "{prompt}",
                            "clip": ["1", 1]
                        }
                    },
                    "3": {
                        "class_type": "EmptyLatentImage",
                        "inputs": {
                            "width": 1024,
                            "height": 1024,
                            "batch_size": 1
                        }
                    },
                    "4": {
                        "class_type": "KSampler",
                        "inputs": {
                            "seed": "{seed}",
                            "steps": 20,
                            "cfg": 8,
                            "sampler_name": "euler",
                            "scheduler": "normal",
                            "denoise": 1,
                            "model": ["1", 0],
                            "positive": ["2", 0],
                            "negative": ["5", 0],
                            "latent_image": ["3", 0]
                        }
                    },
                    "5": {
                        "class_type": "CLIPTextEncode",
                        "inputs": {
                            "text": "blurry, low quality, distorted",
                            "clip": ["1", 1]
                        }
                    },
                    "6": {
                        "class_type": "VAEDecode",
                        "inputs": {
                            "samples": ["4", 0],
                            "vae": ["1", 2]
                        }
                    },
                    "7": {
                        "class_type": "SaveImage",
                        "inputs": {
                            "filename_prefix": "{output_prefix}",
                            "images": ["6", 0]
                        }
                    }
                },
                "default_params": {
                    "steps": 20,
                    "cfg_scale": 8,
                    "width": 1024,
                    "height": 1024
                }
            },
            "technical": {
                "workflow": {
                    "1": {
                        "class_type": "CheckpointLoaderSimple",
                        "inputs": {
                            "ckpt_name": "sdxl_base_1.0.safetensors"
                        }
                    },
                    "2": {
                        "class_type": "CLIPTextEncode",
                        "inputs": {
                            "text": "technical drawing, blueprint, engineering diagram, {prompt}, clean lines, precise, professional",
                            "clip": ["1", 1]
                        }
                    },
                    "3": {
                        "class_type": "EmptyLatentImage",
                        "inputs": {
                            "width": 1024,
                            "height": 1024,
                            "batch_size": 1
                        }
                    },
                    "4": {
                        "class_type": "KSampler",
                        "inputs": {
                            "seed": 42,
                            "steps": 15,
                            "cfg": 7,
                            "sampler_name": "dpmpp_2m",
                            "scheduler": "karras",
                            "denoise": 1,
                            "model": ["1", 0],
                            "positive": ["2", 0],
                            "negative": ["5", 0],
                            "latent_image": ["3", 0]
                        }
                    },
                    "5": {
                        "class_type": "CLIPTextEncode",
                        "inputs": {
                            "text": "colorful, artistic, blurry, organic shapes",
                            "clip": ["1", 1]
                        }
                    },
                    "6": {
                        "class_type": "VAEDecode",
                        "inputs": {
                            "samples": ["4", 0],
                            "vae": ["1", 2]
                        }
                    },
                    "7": {
                        "class_type": "SaveImage",
                        "inputs": {
                            "filename_prefix": "{output_prefix}",
                            "images": ["6", 0]
                        }
                    }
                }
            }
        }

    async def generate_design_render(self, design_blueprint: Dict[str, Any], render_type: str = "photorealistic") -> RenderJob:
        """Generate visual render of the design"""

        job_id = f"render_{design_blueprint['id']}_{render_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting render job: {job_id}")

        # Create render job
        render_job = RenderJob(
            job_id=job_id,
            design_id=design_blueprint["id"],
            render_type=render_type,
            parameters=self._generate_render_parameters(design_blueprint, render_type),
            status="pending",
            output_files=[]
        )

        self.render_jobs[job_id] = render_job

        # Start rendering process
        asyncio.create_task(self._execute_render_job(render_job))

        return render_job

    def _generate_render_parameters(self, design_blueprint: Dict[str, Any], render_type: str) -> Dict[str, Any]:
        """Generate rendering parameters based on design"""

        intent = design_blueprint["intent"]
        components = design_blueprint["components"]

        # Generate descriptive prompt
        category = intent["category"]
        scale = intent["scale"]

        base_prompts = {
            "architecture": f"modern architectural design, {scale} scale, detailed building structure",
            "robotics": f"advanced robotic system, {scale} scale, mechanical components, technical design",
            "vehicle": f"innovative vehicle design, {scale} scale, aerodynamic, engineering blueprint",
            "machine": f"industrial machinery, {scale} scale, mechanical engineering, manufacturing design"
        }

        prompt = base_prompts.get(category, f"engineering design, {scale} scale, technical illustration")

        # Add component details
        component_descriptions = []
        for comp in components[:3]:  # Limit to top 3 components
            component_descriptions.append(f"{comp['name']} component")

        if component_descriptions:
            prompt += f", featuring {', '.join(component_descriptions)}"

        return {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted, amateur, cartoon, anime",
            "seed": 42,
            "steps": 20,
            "cfg_scale": 8,
            "width": 1024,
            "height": 1024,
            "num_images": 1,
            "output_prefix": f"{design_blueprint['id']}_{render_type}"
        }

    async def _execute_render_job(self, render_job: RenderJob):
        """Execute the rendering job using ComfyUI"""

        try:
            render_job.status = "running"

            # Check if ComfyUI is available
            if not await self._check_comfyui_connection():
                logger.warning("ComfyUI not available, using fallback rendering")
                await self._fallback_render(render_job)
                return

            # Generate workflow
            workflow = self._create_comfyui_workflow(render_job)

            # Queue prompt
            prompt_id = await self._queue_comfyui_prompt(workflow)

            if not prompt_id:
                raise Exception("Failed to queue ComfyUI prompt")

            # Wait for completion
            output_files = await self._wait_for_comfyui_completion(prompt_id, render_job.parameters["output_prefix"])

            render_job.status = "completed"
            render_job.output_files = output_files

            logger.info(f"Render job completed: {render_job.job_id}")

        except Exception as e:
            logger.error(f"Render job failed: {render_job.job_id} - {e}")
            render_job.status = "failed"
            await self._fallback_render(render_job)

    async def _check_comfyui_connection(self) -> bool:
        """Check if ComfyUI is running and accessible"""

        try:
            response = requests.get(f"{self.comfyui_endpoint}/system_stats", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _create_comfyui_workflow(self, render_job: RenderJob) -> Dict[str, Any]:
        """Create ComfyUI workflow from template"""

        template = self.templates.get(render_job.render_type, self.templates["photorealistic"])
        workflow = template["workflow"].copy()

        # Update parameters
        params = render_job.parameters

        # Update text prompt
        if "2" in workflow and "inputs" in workflow["2"]:
            workflow["2"]["inputs"]["text"] = params["prompt"]

        # Update seed
        if "4" in workflow and "inputs" in workflow["4"]:
            workflow["4"]["inputs"]["seed"] = params["seed"]
            workflow["4"]["inputs"]["steps"] = params["steps"]
            workflow["4"]["inputs"]["cfg"] = params["cfg_scale"]

        # Update output prefix
        if "7" in workflow and "inputs" in workflow["7"]:
            workflow["7"]["inputs"]["filename_prefix"] = params["output_prefix"]

        return workflow

    async def _queue_comfyui_prompt(self, workflow: Dict[str, Any]) -> Optional[str]:
        """Queue a prompt in ComfyUI"""

        try:
            response = requests.post(
                f"{self.comfyui_endpoint}/prompt",
                json={"prompt": workflow},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("prompt_id")

        except Exception as e:
            logger.error(f"Failed to queue ComfyUI prompt: {e}")

        return None

    async def _wait_for_comfyui_completion(self, prompt_id: str, output_prefix: str, timeout: int = 300) -> List[str]:
        """Wait for ComfyUI job completion and get output files"""

        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                # Check history for completion
                response = requests.get(f"{self.comfyui_endpoint}/history/{prompt_id}", timeout=10)

                if response.status_code == 200:
                    history = response.json()
                    if prompt_id in history:
                        status = history[prompt_id].get("status", {})
                        if status.get("completed", False):
                            # Job completed, find output files
                            return await self._find_output_files(output_prefix)

                await asyncio.sleep(2)  # Wait 2 seconds before checking again

            except Exception as e:
                logger.error(f"Error checking ComfyUI status: {e}")
                await asyncio.sleep(5)

        raise TimeoutError(f"ComfyUI job {prompt_id} timed out")

    async def _find_output_files(self, output_prefix: str) -> List[str]:
        """Find generated output files"""

        # ComfyUI typically saves to output directory
        output_dir = Path("output/renders")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Look for files with the prefix
        files = []
        for file_path in output_dir.glob(f"{output_prefix}*.png"):
            files.append(str(file_path))

        return files

    async def _fallback_render(self, render_job: RenderJob):
        """Fallback rendering when ComfyUI is not available"""

        logger.info(f"Using fallback rendering for job: {render_job.job_id}")

        # Create a simple placeholder image description
        output_dir = Path("output/renders")
        output_dir.mkdir(parents=True, exist_ok=True)

        description_file = output_dir / f"{render_job.parameters['output_prefix']}_description.txt"

        with open(description_file, 'w') as f:
            f.write(f"Render Description for {render_job.design_id}\n")
            f.write(f"Type: {render_job.render_type}\n")
            f.write(f"Prompt: {render_job.parameters['prompt']}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("\nNote: ComfyUI not available. This is a placeholder.\n")
            f.write("To enable AI rendering, start ComfyUI server on localhost:8188\n")

        render_job.output_files = [str(description_file)]
        render_job.status = "completed"

    async def generate_animation_sequence(self, design_blueprint: Dict[str, Any], frames: int = 30) -> RenderJob:
        """Generate animation sequence of the design"""

        job_id = f"anim_{design_blueprint['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        render_job = RenderJob(
            job_id=job_id,
            design_id=design_blueprint["id"],
            render_type="animation",
            parameters={
                "frames": frames,
                "frame_rate": 24,
                "output_prefix": f"{design_blueprint['id']}_anim"
            },
            status="pending",
            output_files=[]
        )

        self.render_jobs[job_id] = render_job

        # For animation, we'd need more complex ComfyUI workflows
        # For now, create placeholder
        asyncio.create_task(self._create_animation_placeholder(render_job))

        return render_job

    async def _create_animation_placeholder(self, render_job: RenderJob):
        """Create animation placeholder"""

        output_dir = Path("output/renders")
        output_dir.mkdir(parents=True, exist_ok=True)

        anim_file = output_dir / f"{render_job.parameters['output_prefix']}.txt"

        with open(anim_file, 'w') as f:
            f.write(f"Animation Sequence for {render_job.design_id}\n")
            f.write(f"Frames: {render_job.parameters['frames']}\n")
            f.write(f"Frame Rate: {render_job.parameters['frame_rate']} fps\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("\nNote: Animation generation requires advanced ComfyUI workflows.\n")

        render_job.output_files = [str(anim_file)]
        render_job.status = "completed"

    def get_render_status(self, job_id: str) -> Optional[RenderJob]:
        """Get status of a render job"""
        return self.render_jobs.get(job_id)

    def get_all_render_jobs(self) -> List[RenderJob]:
        """Get all render jobs"""
        return list(self.render_jobs.values())

    async def batch_render(self, design_blueprints: List[Dict[str, Any]], render_type: str = "photorealistic") -> List[RenderJob]:
        """Batch render multiple designs"""

        jobs = []
        for blueprint in design_blueprints:
            job = await self.generate_design_render(blueprint, render_type)
            jobs.append(job)

            # Small delay to avoid overwhelming ComfyUI
            await asyncio.sleep(0.5)

        return jobs

    def export_render_results(self, job_id: str, output_dir: str = "output/renders") -> str:
        """Export render job results to file"""

        if job_id not in self.render_jobs:
            raise ValueError(f"Render job {job_id} not found")

        job = self.render_jobs[job_id]
        output_file = f"{output_dir}/{job_id}_results.json"

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump({
                "job_id": job.job_id,
                "design_id": job.design_id,
                "render_type": job.render_type,
                "status": job.status,
                "parameters": job.parameters,
                "output_files": job.output_files,
                "created": datetime.now().isoformat()
            }, f, indent=2)

        return output_file