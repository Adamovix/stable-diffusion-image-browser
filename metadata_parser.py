"""
Metadata parser for Stable Diffusion images.
Supports A1111 and ComfyUI metadata formats.
"""
import json
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from typing import Dict, Optional


class SDMetadata:
    """Container for Stable Diffusion metadata."""

    def __init__(self):
        self.model_name: Optional[str] = None
        self.positive_prompt: Optional[str] = None
        self.negative_prompt: Optional[str] = None
        self.seed: Optional[int] = None
        self.steps: Optional[int] = None
        self.cfg_scale: Optional[float] = None
        self.sampler: Optional[str] = None
        self.size: Optional[tuple] = None
        self.loras: list = []  # List of tuples: (lora_name, strength_model)
        self.raw_data: str = ""

    def __str__(self):
        return f"Model: {self.model_name or 'Unknown'}\nPrompt: {self.positive_prompt or 'N/A'}"


def parse_a1111_metadata(params_text: str) -> SDMetadata:
    """Parse A1111 format metadata from parameters text."""
    metadata = SDMetadata()
    metadata.raw_data = params_text

    # A1111 format typically has:
    # Positive prompt
    # Negative prompt: ...
    # Steps: X, Sampler: Y, CFG scale: Z, Seed: W, Size: WxH, Model: M, ...

    lines = params_text.split('\n')

    # Extract positive prompt (usually first line(s) before "Negative prompt:")
    negative_idx = params_text.find('Negative prompt:')
    if negative_idx > 0:
        metadata.positive_prompt = params_text[:negative_idx].strip()
    else:
        # No negative prompt, look for the parameters line
        if len(lines) > 1:
            metadata.positive_prompt = lines[0].strip()
        else:
            metadata.positive_prompt = params_text.strip()

    # Extract negative prompt
    if negative_idx > 0:
        # Find where the parameters start (usually after negative prompt)
        params_match = re.search(r'(Steps:|Sampler:|CFG scale:|Seed:)', params_text[negative_idx:])
        if params_match:
            neg_end = negative_idx + params_match.start()
            negative_text = params_text[negative_idx:neg_end]
            metadata.negative_prompt = negative_text.replace('Negative prompt:', '').strip()
        else:
            # All remaining text is negative prompt
            metadata.negative_prompt = params_text[negative_idx:].replace('Negative prompt:', '').strip()

    # Extract parameters
    # Look for pattern like "Key: Value"
    if 'Steps:' in params_text:
        match = re.search(r'Steps:\s*(\d+)', params_text)
        if match:
            metadata.steps = int(match.group(1))

    if 'Sampler:' in params_text:
        match = re.search(r'Sampler:\s*([^,\n]+)', params_text)
        if match:
            metadata.sampler = match.group(1).strip()

    if 'CFG scale:' in params_text:
        match = re.search(r'CFG scale:\s*([\d.]+)', params_text)
        if match:
            metadata.cfg_scale = float(match.group(1))

    if 'Seed:' in params_text:
        match = re.search(r'Seed:\s*(\d+)', params_text)
        if match:
            metadata.seed = int(match.group(1))

    if 'Size:' in params_text:
        match = re.search(r'Size:\s*(\d+)x(\d+)', params_text)
        if match:
            metadata.size = (int(match.group(1)), int(match.group(2)))

    if 'Model:' in params_text:
        match = re.search(r'Model:\s*([^,\n]+)', params_text)
        if match:
            metadata.model_name = match.group(1).strip()

    # Also check for "Model hash:" format
    if not metadata.model_name and 'Model hash:' in params_text:
        match = re.search(r'Model hash:\s*([^,\n]+)', params_text)
        if match:
            metadata.model_name = f"Hash: {match.group(1).strip()}"

    return metadata


def parse_civitai_metadata(params_text: str) -> SDMetadata:
    """Parse CivitAI format metadata (A1111-like with Civitai resources)."""
    # Start with A1111 parser as base
    metadata = parse_a1111_metadata(params_text)

    # Look for Civitai resources JSON
    if 'Civitai resources:' in params_text:
        try:
            # Extract JSON part after "Civitai resources:"
            resources_idx = params_text.find('Civitai resources:')
            resources_json_start = resources_idx + len('Civitai resources:')

            # Find the end of JSON (look for ], then optional Civitai metadata)
            remaining_text = params_text[resources_json_start:].strip()

            # Try to find where JSON array ends
            json_end = remaining_text.find('], Civitai metadata:')
            if json_end == -1:
                # No metadata section, JSON might end with just ]
                json_str = remaining_text
            else:
                json_str = remaining_text[:json_end + 1]  # Include the ]

            # Parse JSON
            resources = json.loads(json_str)

            # Extract model from checkpoint resource
            for resource in resources:
                if resource.get('type') == 'checkpoint':
                    model_name = resource.get('modelName', '')
                    model_version = resource.get('modelVersionName', '')
                    if model_name:
                        if model_version:
                            metadata.model_name = f"{model_name} ({model_version})"
                        else:
                            metadata.model_name = model_name
                    break

            # Extract LORAs
            loras = []
            for resource in resources:
                if resource.get('type') == 'lora':
                    lora_name = resource.get('modelName', '')
                    weight = resource.get('weight', 1.0)
                    if lora_name:
                        loras.append((lora_name, weight))

            if loras:
                metadata.loras = loras

        except (json.JSONDecodeError, ValueError) as e:
            # If JSON parsing fails, just use the A1111 metadata
            print(f"Error parsing Civitai resources JSON: {e}")
            pass

    return metadata


def parse_comfyui_metadata(workflow_text: str, debug: bool = False) -> SDMetadata:
    """Parse ComfyUI format metadata from workflow JSON."""
    metadata = SDMetadata()
    metadata.raw_data = workflow_text[:500]  # Store first 500 chars

    try:
        data = json.loads(workflow_text)

        if debug:
            print(f"ComfyUI JSON parsed successfully. Found {len(data)} nodes.")

        # ComfyUI stores workflow as a graph of nodes
        # Common node types: KSampler, CheckpointLoaderSimple, CLIPTextEncode, LoraLoader

        # Track text encode nodes to handle positive/negative properly
        text_encode_nodes = []
        lora_nodes = []  # Track LORA information

        # Try to find checkpoint/model and other metadata
        for node_id, node in data.items():
            if not isinstance(node, dict):
                continue

            class_type = node.get('class_type', '')
            inputs = node.get('inputs', {})

            if debug:
                print(f"  Node {node_id}: {class_type}")

            # Checkpoint/Model
            if 'CheckpointLoader' in class_type:
                ckpt = inputs.get('ckpt_name', '')
                if ckpt:
                    metadata.model_name = ckpt
                    if debug:
                        print(f"    Found model: {ckpt}")

            # Text encoding (prompts)
            if 'CLIPTextEncode' in class_type or class_type == 'CLIPTextEncode':
                text = inputs.get('text', '')
                if text:
                    text_encode_nodes.append(text)
                    if debug:
                        print(f"    Found text: {text[:100]}...")

            # LORA loader
            if 'LoraLoader' in class_type or class_type == 'LoraLoader':
                lora_name = inputs.get('lora_name', '')
                strength_model = inputs.get('strength_model', 1.0)
                if lora_name:
                    lora_nodes.append((lora_name, strength_model))
                    if debug:
                        print(f"    Found LORA: {lora_name} (strength: {strength_model})")

            # KSampler settings
            if 'Sampler' in class_type or 'KSampler' in class_type:
                if 'seed' in inputs:
                    metadata.seed = inputs.get('seed')
                if 'steps' in inputs:
                    metadata.steps = inputs.get('steps')
                if 'cfg' in inputs:
                    metadata.cfg_scale = inputs.get('cfg')
                if 'sampler_name' in inputs:
                    metadata.sampler = inputs.get('sampler_name', '')
                if debug:
                    print(f"    Sampler settings: seed={metadata.seed}, steps={metadata.steps}, cfg={metadata.cfg_scale}")

        # Assign text encode nodes to positive/negative
        # Usually first is positive, second is negative
        if len(text_encode_nodes) >= 1:
            metadata.positive_prompt = text_encode_nodes[0]
        if len(text_encode_nodes) >= 2:
            metadata.negative_prompt = text_encode_nodes[1]

        # Assign LORA information
        metadata.loras = lora_nodes

        if debug:
            print(f"Final metadata: model={metadata.model_name}, prompt_len={len(metadata.positive_prompt or '')}, loras={len(lora_nodes)}")

    except json.JSONDecodeError as e:
        if debug:
            print(f"JSON decode error: {e}")
        pass
    except Exception as e:
        if debug:
            print(f"Error parsing ComfyUI metadata: {e}")
        pass

    return metadata


def parse_civitai_workflow(workflow_text: str) -> SDMetadata:
    """
    Parse CivitAI workflow format (URN-based resource references).
    This format uses resource-stack nodes with URN identifiers like:
    urn:air:sdxl:checkpoint:civitai:140272@1240288

    Note: This format often has truncated/malformed JSON due to EXIF size limits.
    We extract what we can using regex patterns.
    """
    metadata = SDMetadata()
    metadata.raw_data = workflow_text[:500]

    try:
        # Extract checkpoint from URN (urn:air:sdxl:checkpoint:civitai:MODEL_ID@VERSION_ID)
        checkpoint_match = re.search(
            r'"ckpt_name"\s*:\s*"urn:air:sdxl:checkpoint:civitai:(\d+)@(\d+)"',
            workflow_text
        )
        if checkpoint_match:
            model_id = checkpoint_match.group(1)
            version_id = checkpoint_match.group(2)
            # We only have IDs, not names - would need API call to resolve
            metadata.model_name = f"CivitAI Model {model_id} (v{version_id})"

        # Extract LORAs with strengths
        lora_matches = re.findall(
            r'"lora_name"\s*:\s*"urn:air:sdxl:lora:civitai:(\d+)@(\d+)"\s*,\s*"strength_model"\s*:\s*([\d.]+)',
            workflow_text
        )
        if lora_matches:
            loras = []
            for model_id, version_id, strength in lora_matches:
                lora_name = f"LORA {model_id}@{version_id}"
                loras.append((lora_name, float(strength)))
            metadata.loras = loras

        # Try to extract prompt text fields
        # Look for "text":"..." patterns
        text_matches = re.findall(r'"text"\s*:\s*"([^"]+)"', workflow_text)
        if text_matches:
            # First text field is usually positive prompt
            if len(text_matches) > 0:
                # Clean up embedding URNs and other noise
                prompt = text_matches[0]
                prompt = re.sub(r'embedding:urn:air:[^,]+,\s*', '', prompt)
                if prompt and len(prompt) > 10:
                    metadata.positive_prompt = prompt

            # Second text field might be negative prompt
            if len(text_matches) > 1:
                neg_prompt = text_matches[1]
                neg_prompt = re.sub(r'embedding:urn:air:[^,]+,\s*', '', neg_prompt)
                if neg_prompt and len(neg_prompt) > 10:
                    metadata.negative_prompt = neg_prompt

        # This format doesn't typically include seed, steps, sampler, etc.
        # Those parameters might be in the workflow but not easily extractable

    except Exception as e:
        # If extraction fails, return what we have
        pass

    return metadata


def extract_metadata(image_path: str) -> Optional[SDMetadata]:
    """
    Extract Stable Diffusion metadata from an image file.
    Supports PNG and JPEG formats with A1111 and ComfyUI metadata.
    """
    try:
        with Image.open(image_path) as img:
            metadata = None

            # PNG files - check all available text chunks
            if img.format == 'PNG':
                info = img.info

                # Debug: print all available keys
                # print(f"PNG info keys for {image_path}: {list(info.keys())}")

                # A1111 format stores in 'parameters'
                if 'parameters' in info:
                    metadata = parse_a1111_metadata(info['parameters'])

                # ComfyUI format - try 'prompt' first (contains execution data), then 'workflow'
                elif 'prompt' in info:
                    metadata = parse_comfyui_metadata(info['prompt'])
                elif 'workflow' in info:
                    metadata = parse_comfyui_metadata(info['workflow'])

                # Check all other text chunks for metadata
                else:
                    for key, value in info.items():
                        if isinstance(value, str):
                            # Try CivitAI format (has Civitai resources JSON)
                            if 'Civitai resources:' in value:
                                metadata = parse_civitai_metadata(value)
                                break
                            # Try A1111 format
                            elif 'Steps:' in value or 'Negative prompt:' in value or 'Sampler:' in value:
                                metadata = parse_a1111_metadata(value)
                                break
                            # Try ComfyUI format or CivitAI workflow format (both JSON)
                            elif value.strip().startswith('{'):
                                # Check if it's CivitAI workflow format (URN-based)
                                if 'urn:air:' in value and 'resource-stack' in value:
                                    metadata = parse_civitai_workflow(value)
                                    break
                                # Otherwise try ComfyUI format
                                else:
                                    try:
                                        json.loads(value)  # Verify it's valid JSON
                                        metadata = parse_comfyui_metadata(value)
                                        break
                                    except json.JSONDecodeError:
                                        # Might be truncated CivitAI workflow - try that parser
                                        if 'urn:air:' in value:
                                            metadata = parse_civitai_workflow(value)
                                            break
                                        pass

                # If no metadata found in PNG text chunks, check EXIF data
                # (CivitAI stores metadata in EXIF user_comment for PNG files)
                if not metadata:
                    exif = img.getexif()
                    if exif:
                        for tag_id, value in exif.items():
                            if isinstance(value, bytes):
                                try:
                                    # CivitAI stores metadata as UTF-16 with 'UNICODE\x00\x00' prefix
                                    if value.startswith(b'UNICODE\x00\x00'):
                                        # Strip the 'UNICODE\x00\x00' prefix and decode as UTF-16-LE
                                        value = value[9:].decode('utf-16-le', errors='ignore')
                                    else:
                                        # Try UTF-8 for other formats
                                        value = value.decode('utf-8', errors='ignore')
                                except:
                                    continue

                            if isinstance(value, str):
                                # Try CivitAI format (has Civitai resources JSON)
                                if 'Civitai resources:' in value:
                                    metadata = parse_civitai_metadata(value)
                                    break
                                # Try to parse as A1111 format
                                elif 'Steps:' in value or 'Negative prompt:' in value or 'Sampler:' in value:
                                    metadata = parse_a1111_metadata(value)
                                    break
                                # Try ComfyUI format or CivitAI workflow format (both JSON)
                                elif value.strip().startswith('{'):
                                    # Check if it's CivitAI workflow format (URN-based)
                                    if 'urn:air:' in value and 'resource-stack' in value:
                                        metadata = parse_civitai_workflow(value)
                                        break
                                    # Otherwise try ComfyUI format
                                    else:
                                        try:
                                            json.loads(value)  # Verify it's valid JSON
                                            metadata = parse_comfyui_metadata(value)
                                            break
                                        except json.JSONDecodeError:
                                            # Might be truncated CivitAI workflow - try that parser
                                            if 'urn:air:' in value:
                                                metadata = parse_civitai_workflow(value)
                                                break
                                            pass

            # JPEG files - check EXIF data more thoroughly
            elif img.format in ['JPEG', 'JPG']:
                exif = img.getexif()
                if exif:
                    # Check EXIF IFD (tag 34665) first - this is where CivitAI stores UserComment
                    try:
                        ifd = exif.get_ifd(34665)
                        if ifd:
                            # Check all IFD tags (especially UserComment - tag 37510)
                            for tag_id, value in ifd.items():
                                if isinstance(value, bytes):
                                    try:
                                        # CivitAI stores metadata as UTF-16 with 'UNICODE\x00\x00' prefix (9 bytes)
                                        if value.startswith(b'UNICODE\x00\x00'):
                                            # Strip the 'UNICODE\x00\x00' prefix (9 bytes) and decode as UTF-16-LE
                                            value = value[9:].decode('utf-16-le', errors='ignore')
                                        else:
                                            # Try UTF-8 for other formats
                                            value = value.decode('utf-8', errors='ignore')
                                    except:
                                        continue

                                if isinstance(value, str):
                                    # Try CivitAI format (has Civitai resources JSON)
                                    if 'Civitai resources:' in value:
                                        metadata = parse_civitai_metadata(value)
                                        break
                                    # Try to parse as A1111 format
                                    elif 'Steps:' in value or 'Negative prompt:' in value or 'Sampler:' in value:
                                        metadata = parse_a1111_metadata(value)
                                        break
                                    # Try ComfyUI format or CivitAI workflow format (both JSON)
                                    elif value.strip().startswith('{'):
                                        # Check if it's CivitAI workflow format (URN-based)
                                        if 'urn:air:' in value and 'resource-stack' in value:
                                            metadata = parse_civitai_workflow(value)
                                            break
                                        # Otherwise try ComfyUI format
                                        else:
                                            try:
                                                json.loads(value)  # Verify it's valid JSON
                                                metadata = parse_comfyui_metadata(value)
                                                break
                                            except json.JSONDecodeError:
                                                # Might be truncated CivitAI workflow - try that parser
                                                if 'urn:air:' in value:
                                                    metadata = parse_civitai_workflow(value)
                                                    break
                                                pass
                    except (KeyError, AttributeError):
                        pass

                    # If no metadata found in IFD, check main EXIF tags
                    if not metadata:
                        for tag_id, value in exif.items():
                            if isinstance(value, bytes):
                                try:
                                    # CivitAI stores metadata as UTF-16 with 'UNICODE\x00\x00' prefix
                                    if value.startswith(b'UNICODE\x00\x00'):
                                        # Strip the 'UNICODE\x00\x00' prefix and decode as UTF-16-LE
                                        value = value[9:].decode('utf-16-le', errors='ignore')
                                    else:
                                        # Try UTF-8 for other formats
                                        value = value.decode('utf-8', errors='ignore')
                                except:
                                    continue

                            if isinstance(value, str):
                                # Try CivitAI format (has Civitai resources JSON)
                                if 'Civitai resources:' in value:
                                    metadata = parse_civitai_metadata(value)
                                    break
                                # Try to parse as A1111 format
                                elif 'Steps:' in value or 'Negative prompt:' in value or 'Sampler:' in value:
                                    metadata = parse_a1111_metadata(value)
                                    break
                                # Try ComfyUI format or CivitAI workflow format (both JSON)
                                elif value.strip().startswith('{'):
                                    # Check if it's CivitAI workflow format (URN-based)
                                    if 'urn:air:' in value and 'resource-stack' in value:
                                        metadata = parse_civitai_workflow(value)
                                        break
                                    # Otherwise try ComfyUI format
                                    else:
                                        try:
                                            json.loads(value)  # Verify it's valid JSON
                                            metadata = parse_comfyui_metadata(value)
                                            break
                                        except json.JSONDecodeError:
                                            # Might be truncated CivitAI workflow - try that parser
                                            if 'urn:air:' in value:
                                                metadata = parse_civitai_workflow(value)
                                                break
                                            pass

            # Store image size if we have metadata
            if metadata:
                metadata.size = img.size

            return metadata

    except Exception as e:
        print(f"Error reading metadata from {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_metadata_summary(metadata: Optional[SDMetadata]) -> Dict[str, str]:
    """Convert metadata to a dictionary for display."""
    if not metadata:
        return {"Status": "No metadata found"}

    summary = {}

    if metadata.model_name:
        summary["Model"] = metadata.model_name
    if metadata.positive_prompt:
        summary["Positive Prompt"] = metadata.positive_prompt
    if metadata.negative_prompt:
        summary["Negative Prompt"] = metadata.negative_prompt
    if metadata.seed is not None:
        summary["Seed"] = str(metadata.seed)
    if metadata.steps is not None:
        summary["Steps"] = str(metadata.steps)
    if metadata.cfg_scale is not None:
        summary["CFG Scale"] = str(metadata.cfg_scale)
    if metadata.sampler:
        summary["Sampler"] = metadata.sampler
    if metadata.size:
        summary["Size"] = f"{metadata.size[0]}x{metadata.size[1]}"
    if metadata.loras:
        # Format LORAs as a readable list
        lora_lines = []
        for lora_name, strength in metadata.loras:
            lora_lines.append(f"  • {lora_name} (strength: {strength})")
        summary["LORAs"] = "\n".join(lora_lines)

    if not summary:
        summary["Raw Data"] = metadata.raw_data[:200] + "..." if len(metadata.raw_data) > 200 else metadata.raw_data

    return summary


def debug_image_metadata(image_path: str):
    """
    Debug utility to print all available metadata in an image file.
    Useful for diagnosing metadata extraction issues.
    """
    print(f"\n=== Debug metadata for: {image_path} ===")

    try:
        with Image.open(image_path) as img:
            print(f"Format: {img.format}")
            print(f"Size: {img.size}")
            print(f"Mode: {img.mode}")

            if img.format == 'PNG':
                print("\nPNG Info (text chunks):")
                info = img.info
                for key, value in info.items():
                    if isinstance(value, str):
                        # Show length and first 200 chars
                        print(f"  {key}: (length: {len(value)} chars)")
                        display_value = value[:200] + "..." if len(value) > 200 else value
                        print(f"    {display_value}")

                        # Try to parse as ComfyUI if it looks like JSON
                        if value.strip().startswith('{'):
                            print(f"\n  Attempting to parse {key} as ComfyUI JSON...")
                            test_metadata = parse_comfyui_metadata(value, debug=True)
                            print()
                    else:
                        print(f"  {key}: <{type(value).__name__}>")

            elif img.format in ['JPEG', 'JPG']:
                print("\nEXIF Data:")
                exif = img.getexif()
                if exif:
                    for tag_id, value in exif.items():
                        # Try to decode bytes
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8', errors='ignore')
                            except:
                                value = f"<bytes: {len(value)} bytes>"

                        # Truncate long strings
                        if isinstance(value, str) and len(value) > 200:
                            print(f"  Tag {tag_id} (0x{tag_id:04x}): (length: {len(value)} chars)")
                            print(f"    {value[:200]}...")
                        else:
                            print(f"  Tag {tag_id} (0x{tag_id:04x}): {value}")
                else:
                    print("  No EXIF data found")

            print("\n=== Extracted metadata (via extract_metadata): ===")
            metadata = extract_metadata(image_path)
            if metadata:
                summary = get_metadata_summary(metadata)
                for key, value in summary.items():
                    # Truncate long values for display
                    if isinstance(value, str) and len(value) > 300:
                        print(f"  {key}: {value[:300]}...")
                    else:
                        print(f"  {key}: {value}")
            else:
                print("  No metadata extracted")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Allow running as a script to debug specific images
    import sys
    if len(sys.argv) > 1:
        for img_path in sys.argv[1:]:
            debug_image_metadata(img_path)
