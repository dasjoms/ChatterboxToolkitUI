# gradio_vc_app.py

# Standard imports
import importlib
import torch
import gradio as gr
from omegaconf import DictConfig
import sys
import os
import re
from datetime import datetime
import time
import soundfile as sf
import logging
import shutil
import zipfile
import json # For manifest
import random

# NLTK for text processing
import nltk

_punkt_is_ready = False
# Define a local NLTK data path within the current working directory
NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')

# Ensure the download directory exists
os.makedirs(NLTK_DATA_PATH, exist_ok=True)

# Add this path to NLTK's data search paths.
# It's crucial this is done *before* any data access or download attempts.
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

try:
    # --- STEP 1: Attempt initial functional test ---
    # Try to use sent_tokenize directly. If it works, 'punkt' is ready.
    # This also acts as a check for any 'punkt_tab' or other internal loading issues.
    nltk.sent_tokenize("This is a simple test sentence for NLTK punkt tokenizer.")
    _punkt_is_ready = True
    print("NLTK 'punkt' tokenizer is functional (already installed and loaded).")

except Exception as e_initial_test: # Catch any error that means sent_tokenize doesn't work
    print(f"NLTK 'punkt' tokenizer is not functional or data is missing/corrupted. Initial test error: {e_initial_test}")
    print(f"Attempting to download/re-download 'punkt' to {NLTK_DATA_PATH} with 'force=True' to ensure completeness...")
    try:
        # `punkt` is the correct package to download. `punkt_tab` is an internal resource name.
        success_download = nltk.download('punkt_tab', download_dir=NLTK_DATA_PATH, quiet=True, force=True)
        
        if success_download:
            # --- STEP 3: Re-test after download ---
            # Immediately try to use it again to confirm functionality after download.
            nltk.sent_tokenize("This is a test sentence after successful download.")
            _punkt_is_ready = True
            print("NLTK 'punkt' tokenizer successfully downloaded and is functional.")
        else:
            raise Exception("NLTK 'punkt' download command returned False.")

    except Exception as e_download:
        # If download or post-download test fails, report the error and mark as not ready.
        print(f"FAILED to make NLTK 'punkt' tokenizer functional after download attempt. Error: {e_download}")
        print("Text splitting functionality will be disabled or might fail.")
        _punkt_is_ready = False # Ensure flag is False if download or post-download test fails.

# End of NLTK setup block. All other existing imports and code remain the same.

# External library imports for models and audio processing
import pydub
import webrtcvad

# NOTE: Make sure these are installed: pip install librosa soundfile huggingface_hub omegaconf numpy nltk pydub webrtcvad-wheels

# --- PORTABILITY FIX: Dynamically find the path to the 'src' directory ---
# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__)) 
# Construct the path to the 'src' directory, which is a sibling to the script
chatterbox_src_path = os.path.join(script_dir, 'src')

# Add the 'src' path to the system path if it's not already there
if chatterbox_src_path not in sys.path:
    sys.path.insert(0, chatterbox_src_path)
    old_sys_path = [] # Define for the print statement below
    print(f"--- Dynamically added '{chatterbox_src_path}' to sys.path ---")
else:
    old_sys_path = [chatterbox_src_path] # Acts as a flag that path was already present


# --- Import and Reload local chatterbox modules ---
import chatterbox.vc
import chatterbox.models.s3gen
import chatterbox.tts
importlib.reload(chatterbox.models.s3gen)
importlib.reload(chatterbox.vc)
importlib.reload(chatterbox.tts)

# --- Debugging print statements (keep for verification) ---
# This print statement is now just for confirming the load path
print(f"Loaded chatterbox.models.s3gen from: {chatterbox.models.s3gen.__file__}")
print(f"Loaded chatterbox.vc from: {chatterbox.vc.__file__}")
print(f"Loaded chatterbox.tts from: {chatterbox.tts.__file__}")
print(f"--------------------")

from chatterbox.vc import ChatterboxVC
from chatterbox.tts import ChatterboxTTS

logging.basicConfig(level=logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model_vc_instance = None
_model_tts_instance = None

# --- Global State for "Regenerate Audio" feature ---
_last_single_tts_output_path_state_value = None # Stores the actual path string

def get_vc_model():
    global _model_vc_instance
    if _model_vc_instance is None:
        print(f"Loading ChatterboxVC model on {DEVICE}...")
        _model_vc_instance = ChatterboxVC.from_pretrained(DEVICE)
        print("ChatterboxVC model loaded successfully.")
    return _model_vc_instance

def get_tts_model():
    global _model_tts_instance
    if _model_tts_instance is None:
        print(f"Loading ChatterboxTTS model on {DEVICE}...")
        _model_tts_instance = ChatterboxTTS.from_pretrained(DEVICE)
        print("ChatterboxTTS model loaded successfully.")
    return _model_tts_instance

def sanitize_filename(name):
    s = str(name)
    s = re.sub(r'[\\/:*?"<>|]+', '', s) 
    s = s.replace(' ', '_') 
    return s

def get_text_snippet_for_filename(text, max_len=30):
    if not text: return "untitled_text"
    s = re.sub(r'[^a-zA-Z0-9\s]', '', text).strip() 
    s = re.sub(r'\s+', '_', s) 
    s = s.strip('_') 
    if not s: return "untitled_text"
    return s[:max_len]

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    # Note: random and numpy are not imported, so these lines would cause an error.
    # If you intend to use them, ensure 'import random' and 'import numpy as np' are at the top.
    # random.seed(seed) 
    # np.random.seed(seed)

global_log_messages_tts = []
global_log_messages_vc = []
global_log_messages_batch_tts = [] 
global_log_messages_batch_vc = []

def yield_tts_updates(log_msg=None, audio_data=None, file_list=None, log_append=True):
    global global_log_messages_tts
    if log_msg:
        if log_append: global_log_messages_tts.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_msg}")
        else: global_log_messages_tts = [f"[{datetime.now().strftime('%H:%M:%S')}] {log_msg}"]
    log_update = gr.update(value="\n".join(global_log_messages_tts))
    if audio_data is not None: 
        audio_update = gr.update(value=audio_data, visible=True)
        files_update = gr.update(value=None, visible=False) 
    elif file_list is not None and len(file_list) > 0: 
        audio_update = gr.update(value=None, visible=False) 
        files_update = gr.update(value=file_list, visible=True)
    else: 
        audio_update = gr.update(value=None, visible=False)
        files_update = gr.update(value=None, visible=False)
    yield log_update, audio_update, files_update

def yield_batch_tts_updates(log_msg=None, file_list=None, log_append=True):
    global global_log_messages_batch_tts 
    if log_msg:
        if log_append: global_log_messages_batch_tts.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_msg}")
        else: global_log_messages_batch_tts = [f"[{datetime.now().strftime('%H:%M:%S')}] {log_msg}"]
    log_update = gr.update(value="\n".join(global_log_messages_batch_tts))
    if file_list is not None and len(file_list) > 0: files_update = gr.update(value=file_list, visible=True)
    else: files_update = gr.update(value=None, visible=False)
    yield log_update, files_update

def yield_batch_vc_updates(log_msg=None, file_list=None, log_append=True):
    global global_log_messages_batch_vc 
    if log_msg:
        if log_append: global_log_messages_batch_vc.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_msg}")
        else: global_log_messages_batch_vc = [f"[{datetime.now().strftime('%H:%M:%S')}] {log_msg}"]
    log_update = gr.update(value="\n".join(global_log_messages_batch_vc))
    if file_list is not None and len(file_list) > 0: files_update = gr.update(value=file_list, visible=True)
    else: files_update = gr.update(value=None, visible=False)
    yield log_update, files_update

def yield_vc_updates(log_msg=None, audio_data=None, file_list=None, log_append=True):
    global global_log_messages_vc
    if log_msg:
        if log_append: global_log_messages_vc.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_msg}")
        else: global_log_messages_vc = [f"[{datetime.now().strftime('%H:%M:%S')}] {log_msg}"]
    log_update = gr.update(value="\n".join(global_log_messages_vc))
    if audio_data is not None: 
        audio_update = gr.update(value=audio_data, visible=True)
        files_update = gr.update(value=None, visible=False) 
    elif file_list is not None and len(file_list) > 0: 
        audio_update = gr.update(value=None, visible=False) 
        files_update = gr.update(value=file_list, visible=True)
    else: 
        audio_update = gr.update(value=None, visible=False)
        files_update = gr.update(value=None, visible=False)
    yield log_update, audio_update, files_update

PROJECTS_BASE_DIR = "projects"
_current_project_root_dir = "" 
_current_project_name = "" 
PROJECT_SUBDIRS = {
    "input_files": "input_files", "voice_conversion": "voice_conversion", "processed_text": "processed_text",
    "processed_audio": "processed_audio", "single_generations_tts": os.path.join("single_generations", "tts"), # Renamed to 'single_generations_tts' for clarity, points to tts
    "single_generations_vc": os.path.join("single_generations", "vc"),
    "batch_generations_tts": os.path.join("batch_generations", "tts"),
    "batch_generations_vc": os.path.join("batch_generations", "vc"),
}
_trigger_ui_update_on_project_state_change = None 

def ensure_projects_base_dir():
    os.makedirs(PROJECTS_BASE_DIR, exist_ok=True)
    logging.info(f"Ensured '{PROJECTS_BASE_DIR}' directory exists.")
def _get_project_path(project_name): return os.path.join(PROJECTS_BASE_DIR, sanitize_filename(project_name))
def list_projects():
    ensure_projects_base_dir()
    projects = [d for d in os.listdir(PROJECTS_BASE_DIR) if os.path.isdir(os.path.join(PROJECTS_BASE_DIR, d)) and not d.startswith('.')] 
    logging.info(f"Found projects: {projects}")
    return projects
def create_project(project_name):
    global _current_project_root_dir, _current_project_name
    if not project_name or not project_name.strip(): raise gr.Error("Project name cannot be empty.")
    sanitized_name = sanitize_filename(project_name)
    project_path = _get_project_path(sanitized_name)
    if os.path.exists(project_path): raise gr.Error(f"Project '{sanitized_name}' already exists.")
    try:
        os.makedirs(project_path)
        for _, subdir_path in PROJECT_SUBDIRS.items(): os.makedirs(os.path.join(project_path, subdir_path), exist_ok=True)
        logging.info(f"Project '{sanitized_name}' created at: {project_path}")
        _current_project_root_dir = project_path
        _current_project_name = sanitized_name
        updated_projects = list_projects() 
        current_dropdown_value = _current_project_name if _current_project_name in updated_projects else (updated_projects[0] if updated_projects else None)
        
        # Main outputs for project selection/creation itself
        main_outputs_tuple = (
            gr.update(choices=updated_projects, value=current_dropdown_value), 
            gr.update(value=_current_project_root_dir), 
            f"Project '{_current_project_name}' created and set as current."
        )
        
        # Get updates for all other dependent UI elements
        other_ui_updates_list = _trigger_ui_update_on_project_state_change() if _trigger_ui_update_on_project_state_change else []
        
        return main_outputs_tuple + tuple(other_ui_updates_list)

    except Exception as e:
        logging.error(f"Error creating project '{sanitized_name}': {e}")
        raise gr.Error(f"Error creating project: {e}")

def set_current_project_instance(project_name=None):
    global _current_project_root_dir, _current_project_name
    if not project_name:
        _current_project_root_dir = ""
        _current_project_name = ""
        logging.info("Current project cleared.")
        main_outputs_tuple = (gr.update(value=None, choices=list_projects()), gr.update(value=""), "No project selected.")
    else: 
        project_path = _get_project_path(project_name)
        if not os.path.isdir(project_path) or not all(os.path.isdir(os.path.join(project_path, PROJECT_SUBDIRS[sub_key])) for sub_key in ["input_files", "voice_conversion", "processed_text", "processed_audio"]):
            logging.error(f"Project directory '{project_name}' not found or incomplete: {project_path}")
            raise gr.Error(f"Project '{project_name}' is not a valid Chatterbox project or is incomplete. Please ensure it contains the expected subfolders (e.g., input_files, voice_conversion).")
        _current_project_root_dir = project_path
        _current_project_name = project_name
        logging.info(f"Current project set to: '{_current_project_name}' at '{_current_project_root_dir}'")
        main_outputs_tuple = (gr.update(value=_current_project_name, choices=list_projects()), gr.update(value=_current_project_root_dir), f"Project '{_current_project_name}' selected.")
        
    other_ui_updates_list = _trigger_ui_update_on_project_state_change() if _trigger_ui_update_on_project_state_change else []
    return main_outputs_tuple + tuple(other_ui_updates_list)

def upload_files_to_project(files, target_folder_key):
    if not _current_project_root_dir: raise gr.Error("No project selected. Please select or create a project first.")
    if not files: return "No files uploaded."
    if target_folder_key not in PROJECT_SUBDIRS: raise gr.Error(f"Invalid target folder key: {target_folder_key}")
    target_dir = os.path.join(_current_project_root_dir, PROJECT_SUBDIRS[target_folder_key])
    os.makedirs(target_dir, exist_ok=True)
    uploaded_count = 0
    messages = []
    for file_obj in files:
        src_path = file_obj.name 
        filename = os.path.basename(src_path)
        dest_path = os.path.join(target_dir, filename)
        try:
            shutil.copy(src_path, dest_path)
            uploaded_count += 1
            logging.info(f"Copied '{src_path}' to '{dest_path}'")
            messages.append(f"Copied: {filename}")
        except Exception as e:
            logging.error(f"Error copying file {filename}: {e}")
            messages.append(f"Failed to copy '{filename}': {e}")
    return f"Successfully copied {uploaded_count} file(s) to '{PROJECT_SUBDIRS[target_folder_key]}'.\n" + "\n".join(messages)
def list_project_files(subdir_key, file_extension_filter=None):
    if not _current_project_root_dir: return []
    target_dir = os.path.join(_current_project_root_dir, PROJECT_SUBDIRS[subdir_key])
    if not os.path.isdir(target_dir): return []
    files = os.listdir(target_dir)
    if file_extension_filter:
        if isinstance(file_extension_filter, str): files = [f for f in files if f.lower().endswith(file_extension_filter)]
        elif isinstance(file_extension_filter, (list, tuple)): files = [f for f in files if any(f.lower().endswith(ext) for ext in file_extension_filter)]
    return sorted(files)
def get_project_file_absolute_path(filename, subdir_key):
    if not _current_project_root_dir: return None
    if subdir_key not in PROJECT_SUBDIRS: return None
    return os.path.join(_current_project_root_dir, PROJECT_SUBDIRS[subdir_key], filename)

def generate_vc(audio_filepath,target_voice_filepath,inference_cfg_rate: float,sigma_min: float,batch_mode: bool,batch_parameter: str,batch_values_str: str ):
    model_vc = get_vc_model()
    yield from yield_vc_updates(log_msg="Starting Voice Conversion...", log_append=False, audio_data=None, file_list=None)
    base_output_dir = ""
    if _current_project_root_dir and _current_project_name:
        yield from yield_vc_updates(log_msg=f"Active project: '{_current_project_name}'. Saving outputs there.")
        if batch_mode: base_output_dir = os.path.join(_current_project_root_dir, PROJECT_SUBDIRS["batch_generations_vc"])
        else: base_output_dir = os.path.join(_current_project_root_dir, PROJECT_SUBDIRS["single_generations_vc"])
    else:
        yield from yield_vc_updates(log_msg="No project selected, using generic 'outputs' folder.")
        base_output_dir = os.path.join("outputs", "voice2voice")
    input_audio_name = os.path.splitext(os.path.basename(audio_filepath))[0] if audio_filepath else "no_input_audio"
    target_voice_name = os.path.splitext(os.path.basename(target_voice_filepath))[0] if target_voice_filepath else "default_voice"
    date_folder = datetime.now().strftime('%Y%m%d')
    vc_base_date_dir = os.path.join(base_output_dir, date_folder)
    vc_pair_dir_name = f"{sanitize_filename(input_audio_name)}_x_{sanitize_filename(target_voice_name)}"
    vc_pair_output_dir = os.path.join(vc_base_date_dir, vc_pair_dir_name)
    os.makedirs(vc_pair_output_dir, exist_ok=True)
    try:
        if batch_mode:
            yield from yield_vc_updates(log_msg="Batch mode enabled. Parsing values...")
            try: batch_values = [float(val.strip()) for val in batch_values_str.split(',') if val.strip()]
            except ValueError: raise gr.Error("Error: Invalid batch values. Please enter comma-separated numbers.")
            if not batch_values: raise gr.Error("Error: Please provide comma-separated values for batch generation.")
            yield from yield_vc_updates(log_msg=f"Batch sweep on '{batch_parameter}' with values: {batch_values}")
            yield from yield_vc_updates(log_msg=f"Generating {len(batch_values)} items in batch...")
            batch_run_dir = os.path.join(vc_pair_output_dir, f"batch_sweep_{sanitize_filename(batch_parameter)}_{datetime.now().strftime('%H%M%S')}")
            os.makedirs(batch_run_dir, exist_ok=True)
            generated_file_paths = []
            for i, value in enumerate(batch_values):
                current_inference_cfg_rate, current_sigma_min, param_prefix = inference_cfg_rate, sigma_min, "unknown"
                if batch_parameter == "Inference CFG Rate": current_inference_cfg_rate, param_prefix = value, f"cfg_{str(value).replace('.', '_')}" 
                elif batch_parameter == "Sigma Min": current_sigma_min, param_prefix = value, f"sigma_{str(value).replace('.', '_')}" 
                else: raise gr.Error(f"Unsupported batch parameter: {batch_parameter}")
                logging.info(f"Generating item {i+1}/{len(batch_values)}: {batch_parameter}={value}") 
                yield from yield_vc_updates(log_msg=f"Generating item {i+1}/{len(batch_values)}: {batch_parameter}={value}")
                wav = model_vc.generate(audio_filepath,target_voice_path=target_voice_filepath,inference_cfg_rate=current_inference_cfg_rate,sigma_min=current_sigma_min)
                output_filename = f"{sanitize_filename(param_prefix)}_{datetime.now().strftime('%H%M%S_%f')}.wav"
                output_path = os.path.join(batch_run_dir, output_filename)
                model_vc.save_wav(wav, output_path)
                generated_file_paths.append(output_path)
                yield from yield_vc_updates(log_msg=f"Saved: {output_path}")
            final_message = f"Batch generation complete. {len(generated_file_paths)} files saved in: {batch_run_dir}"
            yield from yield_vc_updates(log_msg=final_message, file_list=generated_file_paths)
            gr.Info(final_message)
        else: 
            yield from yield_vc_updates(log_msg="Performing single Voice-to-Voice generation...")
            wav = model_vc.generate(audio_filepath,target_voice_path=target_voice_filepath,inference_cfg_rate=inference_cfg_rate,sigma_min=sigma_min)
            single_output_filename = f"output_{datetime.now().strftime('%H%M%S_%f')}.wav"
            single_output_path = os.path.join(vc_pair_output_dir, single_output_filename)
            model_vc.save_wav(wav, single_output_path)
            output_audio_sr_np = (model_vc.sr, wav.squeeze(0).numpy())
            final_message = f"Single Voice-to-Voice generation complete. File saved in: {single_output_path}"
            yield from yield_vc_updates(log_msg=final_message, audio_data=output_audio_sr_np, file_list=[single_output_path])
            gr.Info(final_message)
    except Exception as e:
        error_msg = f"Error during Voice Conversion: {str(e)}"
        yield from yield_vc_updates(log_msg=error_msg, audio_data=None, file_list=None)
        raise gr.Error(error_msg)

def generate_tts(text,audio_prompt_path,exaggeration,temperature,seed_num,cfg_weight,batch_mode: bool,batch_parameter: str,batch_values_str: str):
    global _last_single_tts_output_path_state_value # To store the last generated path
    model_tts = get_tts_model()
    yield from yield_tts_updates(log_msg="Starting Text-to-Voice Generation...", log_append=False, audio_data=None, file_list=None)
    base_output_dir = ""
    if _current_project_root_dir and _current_project_name:
        yield from yield_tts_updates(log_msg=f"Active project: '{_current_project_name}'. Saving outputs there.")
        if batch_mode: base_output_dir = os.path.join(_current_project_root_dir, PROJECT_SUBDIRS["batch_generations_tts"])
        else: base_output_dir = os.path.join(_current_project_root_dir, PROJECT_SUBDIRS["single_generations_tts"])
    else:
        yield from yield_tts_updates(log_msg="No project selected, using generic 'outputs' folder.")
        base_output_dir = os.path.join("outputs", "text2voice")
    date_folder = datetime.now().strftime('%Y%m%d')
    text_snippet_shorter = get_text_snippet_for_filename(text, max_len=15)
    ref_audio_name = sanitize_filename(os.path.splitext(os.path.basename(audio_prompt_path))[0]) if audio_prompt_path else "no_ref_audio"
    combined_tts_folder_name = f"{text_snippet_shorter}_x_{ref_audio_name}"
    tts_output_combined_dir = os.path.join(base_output_dir, date_folder, combined_tts_folder_name)
    os.makedirs(tts_output_combined_dir, exist_ok=True)
    try:
        tts_output_filepath = None # Initialize
        if batch_mode:
            yield from yield_tts_updates(log_msg="Batch mode enabled. Parsing values...")
            batch_values = []
            try:
                if batch_parameter == "Seed": batch_values = [int(val.strip()) for val in batch_values_str.split(',') if val.strip()]
                else: batch_values = [float(val.strip()) for val in batch_values_str.split(',') if val.strip()]
            except ValueError: raise gr.Error("Error: Invalid batch values. Please enter comma-separated numbers.")
            if not batch_values: raise gr.Error("Error: Please provide comma-separated values for batch generation.")
            yield from yield_tts_updates(log_msg=f"Batch sweep on '{batch_parameter}' with values: {batch_values}")
            yield from yield_tts_updates(log_msg=f"Generating {len(batch_values)} items in batch...")
            batch_run_dir = os.path.join(tts_output_combined_dir, f"batch_sweep_{sanitize_filename(batch_parameter)}_{datetime.now().strftime('%H%M%S')}")
            os.makedirs(batch_run_dir, exist_ok=True)
            generated_tts_file_paths = []
            for i, value in enumerate(batch_values):
                current_exaggeration, current_temperature, current_cfg_weight, current_seed_num, param_prefix = exaggeration, temperature, cfg_weight, seed_num, "unknown"
                if batch_parameter == "Exaggeration": current_exaggeration, param_prefix = value, f"exagg_{str(value).replace('.', '_')}"
                elif batch_parameter == "Temperature": current_temperature, param_prefix = value, f"temp_{str(value).replace('.', '_')}"
                elif batch_parameter == "CFG/Pace": current_cfg_weight, param_prefix = value, f"cfg_{str(value).replace('.', '_')}"
                elif batch_parameter == "Seed": current_seed_num, param_prefix = value, f"seed_{value}"
                if current_seed_num != 0: set_seed(int(current_seed_num)); yield from yield_tts_updates(log_msg=f"Using seed: {int(current_seed_num)} for this item.")
                logging.info(f"Generating item {i+1}/{len(batch_values)}: {batch_parameter}={value}") 
                yield from yield_tts_updates(log_msg=f"Generating item {i+1}/{len(batch_values)}: {batch_parameter}={value}")
                wav = model_tts.generate(text,audio_prompt_path=audio_prompt_path,exaggeration=current_exaggeration,temperature=current_temperature,cfg_weight=current_cfg_weight)
                output_sr_np = (model_tts.sr, wav.squeeze(0).numpy())
                tts_output_filename = f"{param_prefix}_{datetime.now().strftime('%H%M%S_%f')}.wav"
                tts_output_filepath = os.path.join(batch_run_dir, tts_output_filename)
                sf.write(tts_output_filepath, output_sr_np[1], output_sr_np[0])
                generated_tts_file_paths.append(tts_output_filepath)
                yield from yield_tts_updates(log_msg=f"Saved: {tts_output_filepath}")
            final_message = f"Batch generation complete. {len(generated_tts_file_paths)} files saved in: {batch_run_dir}"
            yield from yield_tts_updates(log_msg=final_message, file_list=generated_tts_file_paths) 
            gr.Info(final_message)
        else: 
            if seed_num != 0: set_seed(int(seed_num)); yield from yield_tts_updates(log_msg=f"Using seed: {int(seed_num)}")
            wav = model_tts.generate(text,audio_prompt_path=audio_prompt_path,exaggeration=exaggeration,temperature=temperature,cfg_weight=cfg_weight)
            output_sr_np = (model_tts.sr, wav.squeeze(0).numpy())
            tts_output_filename = f"output_{datetime.now().strftime('%H%M%S_%f')}.wav"
            tts_output_filepath = os.path.join(tts_output_combined_dir, tts_output_filename)
            sf.write(tts_output_filepath, output_sr_np[1], output_sr_np[0]) 
            yield from yield_tts_updates(log_msg=f"Saved TTS audio to: {tts_output_filepath}")
            final_message = "Text-to-Voice generation complete."
            yield from yield_tts_updates(log_msg=final_message, audio_data=output_sr_np, file_list=[tts_output_filepath])
            gr.Info(final_message)

        if tts_output_filepath: # If a file was saved (primarily for single mode)
            _last_single_tts_output_path_state_value = tts_output_filepath
            logging.info(f"Updated _last_single_tts_output_path_state_value to: {_last_single_tts_output_path_state_value}")


    except Exception as e:
        error_msg = f"Error during Text-to-Voice generation: {str(e)}"
        _last_single_tts_output_path_state_value = None # Clear on error
        yield from yield_tts_updates(log_msg=error_msg, audio_data=None, file_list=None)
        raise gr.Error(error_msg) 

# --- PERFORMANCE FIX: Set sensible defaults to avoid loading the model at startup ---
default_inference_cfg_rate = 0.5
default_sigma_min = 1e-06
# -----------------------------------------------------------------------------

def toggle_tts_batch_options_visibility(checkbox_state): return gr.update(visible=checkbox_state) 
def toggle_vc_batch_options_visibility(checkbox_state): return gr.update(visible=checkbox_state) 
def update_tts_text_project_dropdown_choices(): return gr.update(choices=list_project_files('processed_text', '.txt'))
def select_tts_text_from_project(filename):
    if not filename: return gr.update(value="")
    path = get_project_file_absolute_path(filename, 'processed_text')
    if not path or not os.path.exists(path): raise gr.Error(f"File not found: {path}")
    with open(path, 'r', encoding='utf-8') as f: content = f.read()
    return gr.update(value=content)

# NEW: Function to select TTS reference audio from project
def select_tts_ref_audio_from_project(filename):
    if not filename: return gr.update(value=None)
    path = get_project_file_absolute_path(filename, 'voice_conversion') # Reference audio for TTS generally comes from voice_conversion for voice cloning
    if not path or not os.path.exists(path): raise gr.Error(f"File not found: {path}")
    return gr.update(value=path)

def update_vc_input_audio_project_dropdown_choices(): return gr.update(choices=list_project_files('processed_audio', '.wav'))
def select_vc_input_audio_from_project(filename):
    if not filename: return gr.update(value=None)
    path = get_project_file_absolute_path(filename, 'processed_audio')
    if not path or not os.path.exists(path): raise gr.Error(f"File not found: {path}")
    return gr.update(value=path)
def update_vc_target_voice_project_dropdown_choices(): return gr.update(choices=list_project_files('voice_conversion', '.wav'))
def select_vc_target_voice_from_project(filename):
    if not filename: return gr.update(value=None)
    path = get_project_file_absolute_path(filename, 'voice_conversion')
    if not path or not os.path.exists(path): raise gr.Error(f"File not found: {path}")
    return gr.update(value=path)

def split_text_document(text_filepath, max_chars_per_chunk):
    yield "Starting text splitting...", gr.update(value=None, visible=False) 
    if not _current_project_root_dir: raise gr.Error("No project selected. Please select or create a project to split text.")
    global _punkt_is_ready 
    if not _punkt_is_ready: raise gr.Error("NLTK 'punkt' tokenizer is not functional. Please check console output during startup for download errors or try restarting the app.")
    output_dir = os.path.join(_current_project_root_dir, PROJECT_SUBDIRS["processed_text"])
    os.makedirs(output_dir, exist_ok=True)
    try:
        if not text_filepath or not os.path.exists(text_filepath): raise gr.Error("No input text file provided or file does not exist.")
        with open(text_filepath, 'r', encoding='utf-8') as f: full_text = f.read()
        sentences = nltk.sent_tokenize(full_text)
        chunks, current_chunk, current_chunk_len = [], [], 0
        original_filename_base = sanitize_filename(os.path.splitext(os.path.basename(text_filepath))[0])
        chunk_count, generated_file_paths = 0, []
        yield f"Loaded text from {os.path.basename(text_filepath)}. Splitting into {len(sentences)} sentences...", gr.update(value=None, visible=False)
        for i, sentence in enumerate(sentences):
            if current_chunk and (current_chunk_len + len(sentence) + (1 if current_chunk else 0) > max_chars_per_chunk):
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text: 
                    chunk_count += 1
                    output_filename = f"{original_filename_base}_part_{chunk_count:03d}.txt"
                    output_path = os.path.join(output_dir, output_filename)
                    with open(output_path, 'w', encoding='utf-8') as outfile: outfile.write(chunk_text)
                    generated_file_paths.append(output_path)
                    yield f"Processed {i+1}/{len(sentences)} sentences. Saved chunk {chunk_count}.", gr.update(value=None, visible=False)
                current_chunk, current_chunk_len = [sentence], len(sentence)
            else:
                current_chunk.append(sentence)
                current_chunk_len += len(sentence) + (1 if current_chunk_len > 0 else 0) 
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunk_count += 1
                output_filename = f"{original_filename_base}_part_{chunk_count:03d}.txt"
                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, 'w', encoding='utf-8') as outfile: outfile.write(chunk_text)
                generated_file_paths.append(output_path)
                yield f"Finished processing sentences. Saved final chunk {chunk_count}.", gr.update(value=None, visible=False)
        if not generated_file_paths: raise gr.Error("No text chunks were generated. The input text might be too short or contains no sentences.")
        final_message = f"Text splitting complete. Generated {len(generated_file_paths)} chunks."
        yield final_message, gr.File(value=generated_file_paths, visible=True)
        gr.Info(final_message)
    except Exception as e:
        error_msg = f"Error during text splitting: {str(e)}"
        yield error_msg, gr.update(value=None, visible=False) 
        raise gr.Error(error_msg)

def _vad_split_audio_segment(audio_segment, frame_rate=16000, frame_duration_ms=30, vad_aggressiveness=3):
    vad = webrtcvad.Vad(vad_aggressiveness)
    audio_segment_resampled = audio_segment.set_frame_rate(frame_rate).set_channels(1).set_sample_width(2)
    samples_per_frame = int(frame_rate * frame_duration_ms / 1000)
    segments, current_segment_start_ms = [], None
    for i in range(0, len(audio_segment_resampled) - samples_per_frame + 1, samples_per_frame):
        frame_start_ms = i * 1000 // frame_rate
        frame_bytes = audio_segment_resampled[frame_start_ms: frame_start_ms + frame_duration_ms].raw_data
        is_speech = vad.is_speech(frame_bytes, frame_rate)
        if is_speech:
            if current_segment_start_ms is None: current_segment_start_ms = frame_start_ms
        elif current_segment_start_ms is not None:
            segments.append((current_segment_start_ms, frame_start_ms + frame_duration_ms)); current_segment_start_ms = None
    if current_segment_start_ms is not None: segments.append((current_segment_start_ms, len(audio_segment_resampled)))
    return segments

def find_best_silence_cut_point(audio_buffer: pydub.AudioSegment, target_ms: int, max_look_back_ms: int, min_chunk_len_ms: int, min_silence_len_for_cut_detection_ms: int):
    search_window_start_ms = max(0, target_ms - max_look_back_ms)
    search_window = audio_buffer[search_window_start_ms:target_ms]
    if not search_window: return target_ms 
    silence_thresh = audio_buffer.dBFS - 10 
    silence_intervals = pydub.silence.detect_silence( search_window, min_silence_len=min_silence_len_for_cut_detection_ms, silence_thresh=silence_thresh)
    best_cut_point_within_buffer = target_ms 
    for silence_start_relative, silence_end_relative in reversed(silence_intervals):
        candidate_cut_point_relative_to_search_window = silence_end_relative 
        absolute_candidate_cut_point = search_window_start_ms + candidate_cut_point_relative_to_search_window
        if absolute_candidate_cut_point >= min_chunk_len_ms:
            best_cut_point_within_buffer = absolute_candidate_cut_point
            break 
    return best_cut_point_within_buffer

def split_audio_document(audio_filepath, max_duration_sec, min_duration_sec, min_silence_for_cut_ms): 
    yield "Starting audio splitting...", gr.update(value=None, visible=False)
    if not _current_project_root_dir: raise gr.Error("No project selected. Please select or create a project to split audio.")
    output_dir = os.path.join(_current_project_root_dir, PROJECT_SUBDIRS["processed_audio"])
    os.makedirs(output_dir, exist_ok=True)
    try:
        if not audio_filepath or not os.path.exists(audio_filepath): raise gr.Error("No input audio file provided or file does not exist.")
        yield f"Loading audio from {os.path.basename(audio_filepath)}...", gr.update(value=None, visible=False)
        audio = pydub.AudioSegment.from_file(audio_filepath)
        original_total_length_ms, max_duration_ms, min_duration_ms = len(audio), max_duration_sec * 1000, min_duration_sec * 1000 
        if min_duration_ms > max_duration_ms: raise gr.Error("Minimum duration (s) cannot be greater than Maximum duration (s).")
        generated_audio_paths, original_filename_base = [], sanitize_filename(os.path.splitext(os.path.basename(audio_filepath))[0])
        speech_segments_vad_results = _vad_split_audio_segment(audio, vad_aggressiveness=0) 
        yield f"Audio loaded ({round(original_total_length_ms/1000, 2)}s). Detected {len(speech_segments_vad_results)} raw speech regions. Resegmenting...", gr.update(value=None, visible=False)
        full_audio_buffer, last_processed_audio_source_ms, chunk_number = pydub.AudioSegment.empty(), 0, 1
        for i, (speech_start, speech_end) in enumerate(speech_segments_vad_results):
            silence_start, silence_end = last_processed_audio_source_ms, speech_start
            if silence_end > silence_start: full_audio_buffer += audio[silence_start:silence_end] 
            full_audio_buffer += audio[speech_start:speech_end]
            last_processed_audio_source_ms = speech_end 
            while True: 
                if len(full_audio_buffer) < min_duration_ms: break 
                target_cut_point_in_buffer = min(len(full_audio_buffer), max_duration_ms)
                look_back_window_ms = min(target_cut_point_in_buffer, max_duration_ms) 
                actual_cut_point_ms = find_best_silence_cut_point(full_audio_buffer, target_ms=target_cut_point_in_buffer, max_look_back_ms=look_back_window_ms, min_chunk_len_ms=min_duration_ms, min_silence_len_for_cut_detection_ms=min_silence_for_cut_ms)
                is_forced_cut = len(full_audio_buffer) > max_duration_ms 
                is_internal_good_cut = (actual_cut_point_ms < len(full_audio_buffer)) and (actual_cut_point_ms >= min_duration_ms)
                if not (is_forced_cut or is_internal_good_cut): break 
                new_chunk_to_export = full_audio_buffer[0:actual_cut_point_ms]
                if len(new_chunk_to_export) < min_duration_ms:
                    yield f"Proposed chunk ({round(len(new_chunk_to_export)/1000, 2)}s) too short, accumulating more.", gr.update(value=None, visible=False)
                    break 
                output_path = os.path.join(output_dir, f"{original_filename_base}_part_{chunk_number:03d}.wav")
                new_chunk_to_export.export(output_path, format="wav")
                generated_audio_paths.append(output_path)
                yield f"Saved audio chunk {chunk_number} (duration: {round(len(new_chunk_to_export)/1000, 2)}s).", gr.update(value=None, visible=False)
                full_audio_buffer = full_audio_buffer[actual_cut_point_ms:] 
                chunk_number += 1
        remaining_audio_after_vad_segments = audio[last_processed_audio_source_ms:]
        full_audio_buffer += remaining_audio_after_vad_segments
        if len(full_audio_buffer) > 0:
            if len(full_audio_buffer) >= min_duration_ms:
                output_path = os.path.join(output_dir, f"{original_filename_base}_part_{chunk_number:03d}.wav")
                full_audio_buffer.export(output_path, format="wav")
                generated_audio_paths.append(output_path)
                yield f"Saved final audio chunk {chunk_number} (duration: {round(len(full_audio_buffer)/1000, 2)}s). All input processed.", gr.update(value=None, visible=False)
            else:
                yield f"Discarding small remaining buffer of {round(len(full_audio_buffer)/1000, 2)}s (less than minimum duration {min_duration_sec}s).", gr.update(value=None, visible=False)
        if not generated_audio_paths: raise gr.Error("No audio chunks were generated. Input may be too short, has no detectable speech, or duration settings are too restrictive.")
        final_message = f"Audio splitting complete. Generated {len(generated_audio_paths)} chunks."
        yield final_message, gr.File(value=generated_audio_paths, visible=True)
        gr.Info(final_message)
    except Exception as e:
        error_msg = f"Error during audio splitting: {str(e)}"
        yield error_msg, gr.update(value=None, visible=False) 
        raise gr.Error(error_msg)

def update_dp_text_project_dropdown_choices(): return gr.update(choices=list_project_files('input_files', '.txt'))
def select_dp_text_from_project(filename):
    if not filename: return gr.update(value=None)
    path = get_project_file_absolute_path(filename, 'input_files')
    if not path or not os.path.exists(path): raise gr.Error(f"File not found: {path}")
    return gr.update(value=path) 
def update_dp_audio_project_dropdown_choices(): return gr.update(choices=list_project_files('input_files', ['.wav', '.mp3', '.flac']))
def select_dp_audio_from_project(filename):
    if not filename: return gr.update(value=None) 
    path = get_project_file_absolute_path(filename, 'input_files')
    if not path or not os.path.exists(path): raise gr.Error(f"File not found: {path}")
    return gr.update(value=path) 

def handle_batch_tts_zip_upload(zip_filepath):
    if not _current_project_root_dir: raise gr.Error("No project selected. Please select or create a project to upload batch files.")
    if not zip_filepath: return gr.update(value="", visible=True), [], "No zip file uploaded."
    extract_dir_name = f"temp_batch_upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    extract_path = os.path.join(_current_project_root_dir, PROJECT_SUBDIRS["input_files"], extract_dir_name)
    os.makedirs(extract_path, exist_ok=True)
    extracted_files = []
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if member.endswith('.txt') and not os.path.basename(member).startswith('__MACOSX'): 
                    target_file_path = os.path.join(extract_path, os.path.basename(member))
                    with zip_ref.open(member) as source, open(target_file_path, "wb") as target: shutil.copyfileobj(source, target)
                    extracted_files.append(target_file_path)
        extracted_files.sort() 
        if not extracted_files: raise gr.Error("No .txt files found in the uploaded zip archive.")
        display_names = [os.path.basename(f) for f in extracted_files]
        message = f"Successfully extracted {len(extracted_files)} text files from zip. Ready for batch TTS."
        return "\n".join(display_names), extracted_files, message 
    except Exception as e:
        if os.path.exists(extract_path): shutil.rmtree(extract_path)
        raise gr.Error(f"Error processing zip file: {e}")
def load_batch_tts_project_files():
    if not _current_project_root_dir: raise gr.Error("No project selected. Please select a project to load files from.")
    txt_files = list_project_files('processed_text', '.txt')
    full_paths = [get_project_file_absolute_path(f, 'processed_text') for f in txt_files]
    if not full_paths: raise gr.Error(f"No .txt files found in '{PROJECT_SUBDIRS['processed_text']}' of the current project.")
    display_names = [os.path.basename(f) for f in full_paths]
    message = f"Loaded {len(full_paths)} text files from project '{_current_project_name}'s '{PROJECT_SUBDIRS['processed_text']}' directory. Ready for batch TTS."
    return "\n".join(display_names), full_paths, message
def clear_batch_tts_files(): return "", [], "File selection cleared."
def update_batch_tts_ref_audio_project_dropdown_choices(): return gr.update(choices=list_project_files('voice_conversion', '.wav'))
def select_batch_tts_ref_audio_from_project(filename):
    if not filename: return gr.update(value=None)
    path = get_project_file_absolute_path(filename, 'voice_conversion')
    if not path or not os.path.exists(path): raise gr.Error(f"File not found: {path}")
    return gr.update(value=path)

def _save_batch_tts_manifest(batch_output_dir, ref_audio_path):
    manifest_path = os.path.join(batch_output_dir, "manifest.json")
    manifest_data = {
        "reference_audio_path": ref_audio_path,
        "batch_run_timestamp": os.path.basename(batch_output_dir),
    }
    try:
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=4)
        logging.info(f"Saved batch TTS manifest to: {manifest_path}")
    except Exception as e:
        logging.error(f"Error saving batch TTS manifest: {e}")

def run_batch_tts(ref_audio_path_tts_batch,tts_exaggeration_batch,tts_temp_batch,tts_seed_num_batch,tts_cfg_weight_batch,text_files_to_process_list,concatenate_output: bool):
    model_tts = get_tts_model()
    yield from yield_batch_tts_updates(log_msg="Starting Batch Text-to-Voice Generation...", log_append=False, file_list=None)
    if not text_files_to_process_list: raise gr.Error("No text files selected for batch processing.")
    if not ref_audio_path_tts_batch or not os.path.exists(ref_audio_path_tts_batch): raise gr.Error("Reference audio for batch TTS is required and must exist.")
    if not _current_project_root_dir: raise gr.Error("No project selected. Outputs cannot be meaningfully saved for batch runs without a project.")
    
    base_output_dir = os.path.join(_current_project_root_dir, PROJECT_SUBDIRS["batch_generations_tts"])
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    batch_run_specific_output_dir = os.path.join(base_output_dir, run_id) # This is the main folder for this specific batch run
    
    single_files_output_dir = os.path.join(batch_run_specific_output_dir, "single_files")
    concatenated_output_dir = os.path.join(batch_run_specific_output_dir, "concatenated")
    
    os.makedirs(single_files_output_dir, exist_ok=True)
    if concatenate_output: os.makedirs(concatenated_output_dir, exist_ok=True)

    # Save manifest
    _save_batch_tts_manifest(batch_run_specific_output_dir, ref_audio_path_tts_batch)

    yield from yield_batch_tts_updates(log_msg=f"Batch run ID: {run_id}. Saving to: {batch_run_specific_output_dir}")
    yield from yield_batch_tts_updates(log_msg=f"Processing {len(text_files_to_process_list)} text files...")
    all_generated_wav_paths = []
    log_messages_list = [] 
    try:
        for i, text_filepath in enumerate(text_files_to_process_list):
            text_filename = os.path.basename(text_filepath)
            with open(text_filepath, 'r', encoding='utf-8') as f: text_content = f.read()
            if tts_seed_num_batch != 0: set_seed(int(tts_seed_num_batch))
            log_messages_list.append(f"Generating audio for {text_filename} ({i+1}/{len(text_files_to_process_list)})...")
            yield from yield_batch_tts_updates(log_msg="\n".join(log_messages_list), log_append=False, file_list=None) 
            wav = model_tts.generate(text_content,audio_prompt_path=ref_audio_path_tts_batch,exaggeration=tts_exaggeration_batch,temperature=tts_temp_batch,cfg_weight=tts_cfg_weight_batch)
            output_sr_np = (model_tts.sr, wav.squeeze(0).numpy())
            output_filename = f"{sanitize_filename(os.path.splitext(text_filename)[0])}_{datetime.now().strftime('%H%M%S_%f')}.wav"
            output_path = os.path.join(single_files_output_dir, output_filename)
            sf.write(output_path, output_sr_np[1], output_sr_np[0])
            all_generated_wav_paths.append(output_path)
            log_messages_list.append(f"-> Saved: {output_filename}")
            yield from yield_batch_tts_updates(log_msg="\n".join(log_messages_list), log_append=False, file_list=None)
        final_files_for_gr_output = all_generated_wav_paths
        if concatenate_output and all_generated_wav_paths:
            yield from yield_batch_tts_updates(log_msg="Concatenating generated audio files...")
            combined_audio = pydub.AudioSegment.silent(duration=100) 
            for wav_path in all_generated_wav_paths:
                combined_audio += pydub.AudioSegment.from_wav(wav_path)
                combined_audio += pydub.AudioSegment.silent(duration=500) 
            concatenated_filename = f"batch_tts_combined_{run_id}.wav"
            concatenated_path = os.path.join(concatenated_output_dir, concatenated_filename)
            combined_audio.export(concatenated_path, format="wav")
            final_files_for_gr_output = [concatenated_path] 
            yield from yield_batch_tts_updates(log_msg=f"Concatenated audio saved to: {concatenated_path}", file_list=final_files_for_gr_output)
            gr.Info("Batch TTS and concatenation complete!")
        else:
            yield from yield_batch_tts_updates(log_msg=f"Batch TTS complete. {len(all_generated_wav_paths)} individual files saved.", file_list=final_files_for_gr_output)
            gr.Info("Batch TTS complete!")
    except Exception as e:
        error_msg = f"Error during Batch Text-to-Voice generation: {str(e)}"
        yield from yield_batch_tts_updates(log_msg=error_msg, file_list=None)
        raise gr.Error(error_msg)

def handle_batch_vc_zip_upload(zip_filepath_vc):
    if not _current_project_root_dir: raise gr.Error("No project selected. Please select or create a project to upload batch files.")
    if not zip_filepath_vc: return gr.update(value="", visible=True), [], "No .zip file uploaded for Batch VC."
    extract_dir_name = f"temp_batch_vc_upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    extract_path = os.path.join(_current_project_root_dir, PROJECT_SUBDIRS["input_files"], extract_dir_name)
    os.makedirs(extract_path, exist_ok=True)
    extracted_files = []
    valid_audio_extensions = ('.wav', '.mp3', '.flac')
    try:
        with zipfile.ZipFile(zip_filepath_vc, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if member.lower().endswith(valid_audio_extensions) and not os.path.basename(member).startswith('__MACOSX'):
                    target_file_path = os.path.join(extract_path, os.path.basename(member))
                    with zip_ref.open(member) as source, open(target_file_path, "wb") as target: shutil.copyfileobj(source, target)
                    extracted_files.append(target_file_path)
        extracted_files.sort()
        if not extracted_files: raise gr.Error(f"No audio files ({', '.join(valid_audio_extensions)}) found in the uploaded zip archive.")
        display_names = [os.path.basename(f) for f in extracted_files]
        message = f"Successfully extracted {len(extracted_files)} audio files from zip. Ready for Batch VC."
        return "\n".join(display_names), extracted_files, message
    except Exception as e:
        if os.path.exists(extract_path): shutil.rmtree(extract_path)
        raise gr.Error(f"Error processing zip file for Batch VC: {e}")
def load_batch_vc_project_files():
    if not _current_project_root_dir: raise gr.Error("No project selected. Please select a project to load files from.")
    audio_files = list_project_files('processed_audio', ('.wav', '.mp3', '.flac'))
    full_paths = [get_project_file_absolute_path(f, 'processed_audio') for f in audio_files]
    if not full_paths: raise gr.Error(f"No audio files found in '{PROJECT_SUBDIRS['processed_audio']}' of the current project.")
    display_names = [os.path.basename(f) for f in full_paths]
    message = f"Loaded {len(full_paths)} audio files from project '{_current_project_name}'s '{PROJECT_SUBDIRS['processed_audio']}' directory. Ready for Batch VC."
    return "\n".join(display_names), full_paths, message
def clear_batch_vc_files(): return "", [], "File selection cleared for Batch VC."
def update_batch_vc_ref_voice_project_dropdown_choices(): return gr.update(choices=list_project_files('voice_conversion', '.wav'))
def select_batch_vc_ref_voice_from_project(filename):
    if not filename: return gr.update(value=None)
    path = get_project_file_absolute_path(filename, 'voice_conversion')
    if not path or not os.path.exists(path): raise gr.Error(f"Reference voice file not found: {path}")
    return gr.update(value=path)
def run_batch_vc(ref_voice_path_vc_batch,vc_inference_cfg_rate_batch,vc_sigma_min_batch,audio_files_to_process_list,concatenate_output_vc: bool):
    model_vc = get_vc_model()
    yield from yield_batch_vc_updates(log_msg="Starting Batch Voice Conversion...", log_append=False, file_list=None)
    if not audio_files_to_process_list: raise gr.Error("No audio files selected for batch VC processing.")
    if not ref_voice_path_vc_batch or not os.path.exists(ref_voice_path_vc_batch): raise gr.Error("Reference voice for batch VC is required and must exist.")
    if not _current_project_root_dir: raise gr.Error("No project selected. Outputs cannot be meaningfully saved for batch runs without a project.")
    base_output_dir = os.path.join(_current_project_root_dir, PROJECT_SUBDIRS["batch_generations_vc"])
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    batch_output_dir = os.path.join(base_output_dir, run_id)
    single_files_output_dir_vc = os.path.join(batch_output_dir, "single_files")
    concatenated_output_dir_vc = os.path.join(batch_output_dir, "concatenated")
    os.makedirs(single_files_output_dir_vc, exist_ok=True)
    if concatenate_output_vc: os.makedirs(concatenated_output_dir_vc, exist_ok=True)
    yield from yield_batch_vc_updates(log_msg=f"Batch VC run ID: {run_id}. Saving to: {batch_output_dir}")
    yield from yield_batch_vc_updates(log_msg=f"Processing {len(audio_files_to_process_list)} audio files...")
    all_converted_wav_paths = []
    log_messages_list_vc = []
    try:
        for i, audio_filepath in enumerate(audio_files_to_process_list):
            audio_filename = os.path.basename(audio_filepath)
            log_messages_list_vc.append(f"Converting voice for {audio_filename} ({i+1}/{len(audio_files_to_process_list)})...")
            yield from yield_batch_vc_updates(log_msg="\n".join(log_messages_list_vc), log_append=False, file_list=None) 
            wav = model_vc.generate(audio_filepath,target_voice_path=ref_voice_path_vc_batch,inference_cfg_rate=vc_inference_cfg_rate_batch,sigma_min=vc_sigma_min_batch)
            output_filename = f"{sanitize_filename(os.path.splitext(audio_filename)[0])}_vc_{datetime.now().strftime('%H%M%S_%f')}.wav"
            output_path = os.path.join(single_files_output_dir_vc, output_filename)
            model_vc.save_wav(wav, output_path)
            all_converted_wav_paths.append(output_path)
            log_messages_list_vc.append(f"-> Saved: {output_filename}")
            yield from yield_batch_vc_updates(log_msg="\n".join(log_messages_list_vc), log_append=False, file_list=None)
        final_files_for_gr_output = all_converted_wav_paths
        if concatenate_output_vc and all_converted_wav_paths:
            yield from yield_batch_vc_updates(log_msg="Concatenating converted audio files...")
            combined_audio = pydub.AudioSegment.silent(duration=100)
            for wav_path in all_converted_wav_paths:
                if os.path.exists(wav_path):
                    audio_segment = pydub.AudioSegment.from_wav(wav_path)
                    combined_audio += audio_segment
                    combined_audio += pydub.AudioSegment.silent(duration=500) 
                else:
                    yield from yield_batch_vc_updates(log_msg=f"Warning: File not found for concatenation: {wav_path}")
            concatenated_filename = f"batch_vc_combined_{run_id}.wav"
            concatenated_path = os.path.join(concatenated_output_dir_vc, concatenated_filename)
            combined_audio.export(concatenated_path, format="wav")
            final_files_for_gr_output = [concatenated_path]
            yield from yield_batch_vc_updates(log_msg=f"Concatenated audio saved to: {concatenated_path}", file_list=final_files_for_gr_output)
            gr.Info("Batch VC and concatenation complete!")
        else:
            yield from yield_batch_vc_updates(log_msg=f"Batch VC complete. {len(all_converted_wav_paths)} individual files saved.", file_list=final_files_for_gr_output)
            gr.Info("Batch VC complete!")
    except Exception as e:
        error_msg = f"Error during Batch Voice Conversion: {str(e)}"
        yield from yield_batch_vc_updates(log_msg=error_msg, file_list=None)
        raise gr.Error(error_msg)

# --- Data Preparation Helpers for Editing ---
def update_edit_text_dropdown_choices():
    new_choices = list_project_files('processed_text', '.txt')
    # Return two values: an update for the dropdown, and the list of choices for the state
    return gr.update(choices=new_choices, value=None), new_choices 

def load_edit_text_content_only(filename): # Renamed for clarity
    """Loads content of selected text file into the editor, enables save button."""
    if not _current_project_root_dir:
        # This case should ideally be handled by UI visibility, but a last-resort check
        return "", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), "No project selected.", [] # Return default states
    if not filename:
        # No file selected in dropdown, reset editor state
        return "", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), "Select a text file from the dropdown to edit.", []
    
    path = get_project_file_absolute_path(filename, 'processed_text')
    if not path or not os.path.exists(path):
        return "", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), f"Error: File '{filename}' not found. Please refresh list.", []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fetch all available files for navigation from the project subdirectory
        all_available_files = list_project_files('processed_text', '.txt')
        nav_buttons_interactive = len(all_available_files) > 1

        # Return content, interactiveness for save button, and log message
        return (
            gr.update(value=content, interactive=True), 
            gr.update(interactive=True), 
            # Correctly set interactive state for prev/next buttons
            gr.update(interactive=nav_buttons_interactive), # Prev button
            gr.update(interactive=nav_buttons_interactive), # Next button
            f"Loaded '{filename}' for editing. You can now make changes.",
            all_available_files # Update the state with full list
        )
    except Exception as e:
        return "", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), f"Error loading '{filename}': {e}", []

def save_edited_text_content(filename_selected_in_dropdown, new_content_from_textbox):
    """Saves the edited content back to the original file."""
    if not _current_project_root_dir:
        raise gr.Error("No project selected. Save failed.")
    if not filename_selected_in_dropdown:
        raise gr.Error("No file selected for saving.")

    path = get_project_file_absolute_path(filename_selected_in_dropdown, 'processed_text')
    if not path or not os.path.exists(path):
        raise gr.Error(f"Original file not found: {path}. Cannot save.")

    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content_from_textbox)
        
        return gr.update(value=new_content_from_textbox, interactive=True), gr.update(interactive=True), f"Changes to '{filename_selected_in_dropdown}' saved successfully."
    except Exception as e:
        raise gr.Error(f"Error saving file '{filename_selected_in_dropdown}': {e}")

def navigate_text_file(current_filename: str, file_list_state: list, direction: int):
    """Navigates to the next/previous file in the list."""
    if not file_list_state: 
        # Note: The dropdown value might not be None here, it might be the last selected value.
        # If the list is empty, always disable buttons and give a message.
        return None, "No files to navigate.", gr.update(interactive=False), gr.update(interactive=False) 
        
    try:
        # Find current index. If not found (e.g., file deleted), default to a valid index or start/end.
        current_idx = file_list_state.index(current_filename) if current_filename in file_list_state else -1
        
        if current_idx == -1 and len(file_list_state) > 0:
            # If current file isn't in list but list is not empty, start from beginning/end
            new_idx = 0 if direction == 1 else len(file_list_state) - 1
        else:
            # Normal navigation
            new_idx = (current_idx + direction) % len(file_list_state) 

        next_filename = file_list_state[new_idx]
        
        # Navigation buttons are interactive if there's more than one file to navigate between
        nav_buttons_interactive = len(file_list_state) > 1

        return next_filename, f"Navigating to: {next_filename}", gr.update(interactive=nav_buttons_interactive), gr.update(interactive=nav_buttons_interactive)
    except Exception as e:
        # Fallback on unexpected error, keep current filename, disable nav, log error
        return current_filename, f"Error navigating files: {e}", gr.update(interactive=False), gr.update(interactive=False)

def load_random_text_file():
    if not _current_project_root_dir:
        return None, gr.update(value="", interactive=False), gr.update(interactive=False), gr.update(interactive=False), "No project selected.", []
        
    # Get all potential files to display
    all_txt_files = list_project_files('processed_text', '.txt')
    if not all_txt_files:
        return None, gr.update(value="", interactive=False), gr.update(interactive=False), gr.update(interactive=False), "No .txt files found in project's processed_text folder.", []
        
    # Select a random file
    random_filename = random.choice(all_txt_files)
    
    # Load its content
    file_path = get_project_file_absolute_path(random_filename, 'processed_text')
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        nav_buttons_interactive = len(all_txt_files) > 1

        return (
            random_filename, # dropdown value
            gr.update(value=content, interactive=True), # textbox value
            gr.update(interactive=nav_buttons_interactive), # Prev button
            gr.update(interactive=nav_buttons_interactive), # Next button
            f"Loaded random file: '{random_filename}'.", # Log message
            all_txt_files # Update current_editable_files_state
        )
    except Exception as e:
        return None, gr.update(value="", interactive=False), gr.update(interactive=False), gr.update(interactive=False), f"Error loading random file '{random_filename}': {e}", []


# --- Regenerate Audio Tab Helpers ---
def list_batch_tts_runs():
    if not _current_project_root_dir:
        return []
    batch_tts_base = os.path.join(_current_project_root_dir, PROJECT_SUBDIRS["batch_generations_tts"])
    if not os.path.isdir(batch_tts_base):
        return []
    runs = [d for d in os.listdir(batch_tts_base) if os.path.isdir(os.path.join(batch_tts_base, d))]
    return sorted(runs, reverse=True) # Newest first

def _load_batch_tts_manifest(batch_run_dir_name):
    if not _current_project_root_dir or not batch_run_dir_name:
        return None
    manifest_path = os.path.join(_current_project_root_dir, PROJECT_SUBDIRS["batch_generations_tts"], batch_run_dir_name, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading manifest {manifest_path}: {e}")
    return None

def load_batch_tts_run_files(batch_run_dir_name):
    if not _current_project_root_dir or not batch_run_dir_name:
        # If no batch run selected or project not active, reset all related UI elements
        return (
            gr.update(choices=[], value=None, interactive=False), # Audio file dropdown
            "", # Log message
            None, # _regen_selected_batch_run_path_state
            None, # _regen_batch_ref_audio_path_state
            gr.update(interactive=False), # regen_send_to_tts_btn
            gr.update(interactive=False, visible=True), # regen_concatenate_btn
            gr.update(value=None, visible=False) # regen_concatenated_output_file
        )

    selected_run_full_path = os.path.join(_current_project_root_dir, PROJECT_SUBDIRS["batch_generations_tts"], batch_run_dir_name)
    single_files_dir = os.path.join(selected_run_full_path, "single_files")

    new_regen_selected_batch_run_path_state = selected_run_full_path

    audio_files = []
    if os.path.isdir(single_files_dir):
        audio_files = sorted([f for f in os.listdir(single_files_dir) if f.lower().endswith('.wav')])
    
    # Determine interactivity based on whether audio files were found
    is_interactive = bool(audio_files)

    manifest_data = _load_batch_tts_manifest(batch_run_dir_name)
    new_regen_batch_ref_audio_path_state = None
    if manifest_data and "reference_audio_path" in manifest_data:
        new_regen_batch_ref_audio_path_state = manifest_data["reference_audio_path"]
        log_msg = f"Loaded {len(audio_files)} audio files from batch run '{batch_run_dir_name}'. Ref audio: {os.path.basename(new_regen_batch_ref_audio_path_state if new_regen_batch_ref_audio_path_state else 'N/A')}"
    else:
        log_msg = f"Loaded {len(audio_files)} audio files from batch run '{batch_run_dir_name}'. Manifest or reference audio not found."
        gr.Warning("Manifest for selected batch run is missing or does not contain reference audio path. 'Send to TTS' might not work correctly for reference voice.")

    # Modified return to explicitly set interactive state for the dropdown
    # Also ensure regen_send_to_tts_btn is re-disabled here, as a new batch run selection clears the sub-selection.
    return (
        gr.update(choices=audio_files, value=None, interactive=is_interactive), # Updated here
        log_msg,
        new_regen_selected_batch_run_path_state,
        new_regen_batch_ref_audio_path_state,
        gr.update(interactive=False), # Disable 'Send to TTS' button when a new run is selected
        gr.update(interactive=is_interactive, visible=True), # Enable/disable concatenate button
        gr.update(value=None, visible=False) # Hide/clear concatenated file output
    )

def load_selected_batch_audio_for_preview(selected_audio_filename, selected_batch_run_path_state):
    if not selected_audio_filename or not selected_batch_run_path_state:
        return gr.update(value=None, label="Preview"), "No audio file selected.", None, None, gr.update(interactive=False)

    audio_file_full_path = os.path.join(selected_batch_run_path_state, "single_files", selected_audio_filename)
    
    # Derive original text file name base
    # Filename: {original_base}_{timestamp}.wav
    # Example: my_text_part_001_234859_123456.wav -> my_text_part_001.txt
    
    # Remove the .wav extension first
    base_name_no_ext = os.path.splitext(selected_audio_filename)[0]
    
    # Split by underscore. The timestamp part is typically the last two segments if they are numeric.
    parts = base_name_no_ext.split('_')
    original_text_filename_base = ""

    # Heuristic to find the original text file name by removing the timestamp
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit() and len(parts[-2]) == 6: # Heuristic for HHMMS_millis
        original_text_filename_base = "_".join(parts[:-2])
    elif len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) > 3: # Heuristic for _timestamp_millis
        original_text_filename_base = "_".join(parts[:-1])
    else: # Fallback: assume everything before the last underscore
        original_text_filename_base = parts[0] if len(parts) == 1 else "_".join(parts[:-1])
        logging.warning(f"Could not robustly determine original text base from '{selected_audio_filename}'. Using fallback: '{original_text_filename_base}'")

    original_text_filename_with_ext = original_text_filename_base + ".txt"
    
    # Try to find this in the project's 'processed_text' folder
    original_txt_path = get_project_file_absolute_path(original_text_filename_with_ext, 'processed_text')

    if not os.path.exists(audio_file_full_path):
        return gr.update(value=None, label="Preview Error"), f"Audio file not found: {audio_file_full_path}", None, None, gr.update(interactive=False)

    if not original_txt_path or not os.path.exists(original_txt_path):
        gr.Warning(f"Original text file '{original_text_filename_with_ext}' not found in processed_text. 'Send to TTS' may not load text.")
        original_txt_path = None # Ensure it's None if not found

    return gr.update(value=audio_file_full_path, label=f"Preview: {selected_audio_filename}"), \
        f"Selected: {selected_audio_filename}. Original text suspected: {original_text_filename_with_ext}", \
        audio_file_full_path, \
        original_txt_path, \
        gr.update(interactive=True) # Enable Send to TTS button

def action_send_to_single_tts(original_txt_path, batch_ref_audio_path_state):
    if not original_txt_path or not os.path.exists(original_txt_path):
        raise gr.Error("Original text file path is not set or file does not exist. Cannot send to TTS.")
    if not batch_ref_audio_path_state or not os.path.exists(batch_ref_audio_path_state):
        raise gr.Error("Reference audio for the batch run is not set or file does not exist. Cannot send to TTS.")

    with open(original_txt_path, 'r', encoding='utf-8') as f:
        text_content = f.read()
    
    log_message = f"Sent '{os.path.basename(original_txt_path)}' and ref audio '{os.path.basename(batch_ref_audio_path_state)}' to Single TTS tab."
    
    # Return updates for: main_tabs, single_gen_tabs, tts_text, tts_ref_audio, regen_status_log, regen_replace_btn
    return gr.update(selected="single_generation_main_tab"), \
        gr.update(selected="tts_sub_tab"), \
        gr.update(value=text_content), \
        gr.update(value=batch_ref_audio_path_state), \
        log_message, \
        gr.update(interactive=True) # Enable replace button

def action_replace_batch_audio(selected_batch_audio_to_replace, _last_single_tts_output_path_val_from_state):
    global _last_single_tts_output_path_state_value # Access the global python variable
    
    if not selected_batch_audio_to_replace:
        raise gr.Error("No batch audio file selected in the 'Regenerate' tab to replace.")
    
    # Use the value from the Gradio state, which should be updated by the get_last_single_tts_path chain
    last_generated_path = _last_single_tts_output_path_val_from_state

    if not last_generated_path or not os.path.exists(last_generated_path):
        raise gr.Error(f"No recent single TTS generation found ('{last_generated_path}') to use for replacement, or the file path is invalid.")

    try:
        shutil.copy2(last_generated_path, selected_batch_audio_to_replace)
        log_message = f"Successfully replaced '{os.path.basename(selected_batch_audio_to_replace)}' with '{os.path.basename(last_generated_path)}'."
        gr.Info(log_message)
        
        # Clear the global Python state after successful replacement
        _last_single_tts_output_path_state_value = None
        
        # Return new preview for the replaced file, log message, and disable replace button again
        return gr.update(value=selected_batch_audio_to_replace), log_message, gr.update(interactive=False), None # Clearing gr.State
    except Exception as e:
        error_msg = f"Error replacing audio file: {e}"
        gr.Error(error_msg)
        return gr.update(), error_msg, gr.update(interactive=True), last_generated_path # Keep button interactive on error, keep state

def concatenate_batch_run_audios(batch_run_path_state):
    """
    Concatenates all audio files from a selected batch run's 'single_files' directory.
    """
    if not batch_run_path_state or not os.path.isdir(batch_run_path_state):
        raise gr.Error("No valid batch run selected. Cannot concatenate.")

    single_files_dir = os.path.join(batch_run_path_state, "single_files")
    if not os.path.isdir(single_files_dir):
        raise gr.Error(f"Could not find the 'single_files' directory in the selected batch run: {batch_run_path_state}")

    # Get a sorted list of all WAV files to ensure correct order
    audio_files_to_concat = sorted([
        os.path.join(single_files_dir, f) for f in os.listdir(single_files_dir)
        if f.lower().endswith('.wav')
    ])

    if not audio_files_to_concat:
        raise gr.Error("No .wav files found in the 'single_files' directory to concatenate.")

    try:
        log_message = f"Starting concatenation for {len(audio_files_to_concat)} files in batch run '{os.path.basename(batch_run_path_state)}'..."
        gr.Info(log_message)

        # Concatenation logic (re-used from run_batch_tts)
        combined_audio = pydub.AudioSegment.silent(duration=100)
        for wav_path in audio_files_to_concat:
            combined_audio += pydub.AudioSegment.from_wav(wav_path)
            combined_audio += pydub.AudioSegment.silent(duration=100) # Add silence between clips

        # Define output directory and path
        concatenated_output_dir = os.path.join(batch_run_path_state, "concatenated")
        os.makedirs(concatenated_output_dir, exist_ok=True)
        
        # Create a unique filename
        run_name = os.path.basename(batch_run_path_state)
        concatenated_filename = f"concatenated_from_regen_{run_name}_{datetime.now().strftime('%H%M%S')}.wav"
        concatenated_path = os.path.join(concatenated_output_dir, concatenated_filename)

        combined_audio.export(concatenated_path, format="wav")

        final_message = f"Concatenation complete. File saved to: {concatenated_path}"
        gr.Info(final_message)
        
        # Return the status message and the path to the newly created file
        return final_message, gr.update(value=concatenated_path, visible=True)

    except Exception as e:
        error_msg = f"Error during audio concatenation: {str(e)}"
        gr.Error(error_msg)
        return error_msg, gr.update(value=None, visible=False)

# --- Project File Management Functions ---
def delete_project_items_backend(selected_paths, current_project_root_dir_val):
    if not current_project_root_dir_val or not os.path.isdir(current_project_root_dir_val):
        return "No active project detected for deletion.", gr.update(root_dir="") 
    if not selected_paths:
        return "No items selected for deletion.", gr.update(root_dir=current_project_root_dir_val) # Refresh explorer even if nothing deleted

    messages = []
    project_base_abs_path = os.path.abspath(current_project_root_dir_val)

    # Convert selected_paths to absolute paths for robust checking
    abs_selected_paths = [os.path.abspath(p) for p in selected_paths]

    # Safety check: Ensure all paths are within the current project directory and not the project root itself
    for path in abs_selected_paths:
        # Check if path starts with project_base_abs_path, ensuring it's not outside
        if not path.startswith(project_base_abs_path + os.sep): # Add os.sep to prevent partial matches like /projects/myproject_temp
            messages.append(f"SECURITY ALERT: Attempted to delete item outside project directory: {path}. Operation aborted for this item.")
        # Check against project root and general projects base directory
        elif path == project_base_abs_path or path == os.path.abspath(PROJECTS_BASE_DIR):
            messages.append(f"SECURITY ALERT: Cannot delete the main project directory itself or the projects base directory: {path}. Operation aborted.")
        else:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    messages.append(f"Deleted folder: {os.path.basename(path)}")
                elif os.path.isfile(path):
                    os.remove(path)
                    messages.append(f"Deleted file: {os.path.basename(path)}")
                else: # Should not happen if path exists and is valid
                    messages.append(f"Skipped (not found or neither file nor dir): {os.path.basename(path)}")
            except OSError as e:
                messages.append(f"Error deleting {os.path.basename(path)}: {e}")

    # Re-render the file explorer by setting its root_dir to itself (forces update)
    # The value is also cleared since the selected files are deleted.
    return "\n".join(messages), gr.update(root_dir=current_project_root_dir_val, value=[])

def refresh_project_explorer_root_update(current_project_root_dir_val_from_state):
    # This simply returns the current_project_root_dir to update the FileExplorer's root_dir.
    # Gradio's FileExplorer updates its view when its root_dir output changes, even to the same value.
    file_explorer_root = current_project_root_dir_val_from_state if current_project_root_dir_val_from_state else ""
    return gr.update(root_dir = file_explorer_root, interactive=bool(file_explorer_root), value=[])

def create_zip_from_selection(selected_paths, project_root_dir):
    """
    Creates a zip file from the selected files and folders in the FileExplorer.
    """
    if not project_root_dir:
        raise gr.Error("No project selected. Cannot create download.")
    if not selected_paths:
        return "No files or folders selected for download.", gr.update(visible=False)

    temp_dir = "temp_downloads"
    os.makedirs(temp_dir, exist_ok=True)
    
    project_name = sanitize_filename(os.path.basename(project_root_dir))
    zip_filename = f"{project_name}_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = os.path.join(temp_dir, zip_filename)

    added_files_set = set()
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for path in selected_paths:
                # The path inside the zip should be relative to the project root directory
                arcname_base = os.path.relpath(path, project_root_dir)
                
                if os.path.isfile(path):
                    if path not in added_files_set:
                        zipf.write(path, arcname_base)
                        added_files_set.add(path)
                elif os.path.isdir(path):
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            full_file_path = os.path.join(root, file)
                            if full_file_path not in added_files_set:
                                arcname = os.path.relpath(full_file_path, project_root_dir)
                                zipf.write(full_file_path, arcname)
                                added_files_set.add(full_file_path)

        if not added_files_set:
            return "No valid files found in selection to create a zip.", gr.update(visible=False, value=None)

        message = f"Successfully created zip archive with {len(added_files_set)} files. Ready for download."
        gr.Info(message)
        return message, gr.update(value=zip_path, visible=True)

    except Exception as e:
        error_msg = f"Error creating zip file: {e}"
        gr.Error(error_msg)
        return error_msg, gr.update(visible=False)


# --- Gradio Interface Layout ---
with gr.Blocks(title="ChatterboxToolkitUI") as demo:
    gr.Markdown(
        """
        # ChatterboxToolkitUI
        <p style="text-align: left; font-size: 1.1em;">
        Your comprehensive platform for audio generation and manipulation.
        </p>
        """
    )
    # --- Main Tabs ---
    with gr.Tabs(elem_id="main_tabs_id") as main_tabs: # Name the main tabs for better control
        # --- Main Tab 1: Single Generation ---
        with gr.Tab("Single Generation", id="single_generation_main_tab"):
            with gr.Tabs(elem_id="single_generation_sub_tabs_id") as single_generation_sub_tabs: # Nested Tabs for Single Generation modes
                # --- Tab 1.1: Text to Voice ---
                with gr.Tab("Text to Voice", id="tts_sub_tab") as tts_tab:
                    gr.Markdown(
                        """
                        ## Text to Voice Synthesis (TTS)
                        Convert written text into natural-sounding speech.
                        """
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1): # Left column for inputs
                            gr.Markdown("### Input Text")
                            tts_text = gr.Textbox(
                                value="The quick brown fox jumps over the lazy dog.",
                                label="Text to synthesize (max chars 300)",
                                max_lines=5,
                                interactive=True,
                            )
                            with gr.Row(visible=False) as tts_text_project_row: # Hidden by default, controls visibility of its children
                                tts_load_text_from_project_btn = gr.Button("Load Text from Project") # Visible on project active
                                tts_text_project_dropdown = gr.Dropdown(label="Select Text File from Project", visible=False, allow_custom_value=False) # Removed choices=[]
                                tts_text_project_refresh_btn = gr.Button("Refresh", visible=False)

                            gr.Markdown("### Reference Audio (Speaker Voice) (Max. 40s)")
                            tts_ref_audio = gr.Audio(
                                sources=["upload", "microphone"], 
                                type="filepath", 
                                label="Upload or record audio to define the target speaker's voice.", 
                                interactive=True
                            )
                            with gr.Row(visible=False) as tts_ref_audio_project_row: # Hidden by default, controls visibility of its children
                                tts_load_ref_audio_from_project_btn = gr.Button("Load Reference Audio from Project") # Visible on project active
                                tts_ref_audio_project_dropdown = gr.Dropdown(label="Select Reference Audio from Project", visible=False, allow_custom_value=False) # Removed choices=[]
                                tts_ref_audio_project_refresh_btn = gr.Button("Refresh", visible=False)


                            gr.Markdown("### TTS Parameters")
                            tts_exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration", info="Neutral = 0.5, extreme values can be unstable.", value=.5)
                            tts_cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", info="Classifier-Free Guidance weight for pacing and clarity.", value=0.5)

                            with gr.Accordion("More TTS Options", open=False):
                                tts_seed_num = gr.Number(value=0, label="Random seed (0 for random)", info="Set a seed for reproducible generations (0 for truly random).")
                                tts_temp = gr.Slider(0.05, 5, step=.05, label="Temperature", info="Controls the randomness of the output speech, higher values for more variability.", value=.8)

                            # TTS Batch Configuration Section (collapsible)
                            with gr.Accordion("Batch Parameter Sweep Options", open=False) as tts_batch_accordion:
                                tts_batch_mode_checkbox = gr.Checkbox(label="Enable Batch Parameter Sweep", value=False, interactive=True)
                                with gr.Column(visible=False) as tts_batch_options_group:
                                    gr.Markdown("#### Batch Configuration")
                                    tts_batch_parameter_dropdown = gr.Dropdown(
                                        choices=["Exaggeration", "CFG/Pace", "Temperature", "Seed"],
                                        label="Parameter to Vary in Batch", 
                                        value="Exaggeration", 
                                        interactive=True
                                    )
                                    tts_batch_values_textbox = gr.Textbox(
                                        label="Comma-separated Values for Batch Sweep", 
                                        placeholder="e.g., 0.5, 0.7, 0.9 or 1, 10, 100",
                                        interactive=True
                                    )
                                    
                            # Link checkbox to visibility of batch options group
                            tts_batch_mode_checkbox.change(
                                fn=toggle_tts_batch_options_visibility,
                                inputs=[tts_batch_mode_checkbox],
                                outputs=[tts_batch_options_group]
                            )

                            tts_run_btn = gr.Button("🎤 Synthesize Speech", variant="primary")

                        with gr.Column(scale=1): # Right column for logs and output
                            gr.Markdown("### TTS Log")
                            tts_log = gr.Textbox(label="Log", lines=10, interactive=False)
                            gr.Markdown("### Synthesized Audio Output")
                            tts_audio_output = gr.Audio(label="Playback", type="numpy", visible=False)
                            tts_output_files = gr.File(label="Download Generated Audio(s)", visible=False)

                    # TTS Button Click Event
                    tts_run_btn.click(
                        fn=generate_tts,
                        inputs=[
                            tts_text,
                            tts_ref_audio,
                            tts_exaggeration,
                            tts_temp,
                            tts_seed_num,
                            tts_cfg_weight,
                            tts_batch_mode_checkbox,
                            tts_batch_parameter_dropdown,
                            tts_batch_values_textbox,
                        ],
                        outputs=[
                            tts_log,
                            tts_audio_output,
                            tts_output_files
                        ],
                        show_progress='minimal'
                    )

                    # TTS Project-specific UI Interactions
                    tts_load_text_from_project_btn.click(
                        fn=lambda: gr.update(value=""), outputs=[tts_text]
                    ).then(
                        fn=update_tts_text_project_dropdown_choices, outputs=[tts_text_project_dropdown]
                    )
                    tts_text_project_dropdown.change(select_tts_text_from_project, inputs=[tts_text_project_dropdown], outputs=[tts_text])
                    tts_text_project_refresh_btn.click(update_tts_text_project_dropdown_choices, outputs=[tts_text_project_dropdown])

                    tts_load_ref_audio_from_project_btn.click(
                        fn=lambda: gr.update(value=None), outputs=[tts_ref_audio]
                    ).then(
                        fn=update_batch_tts_ref_audio_project_dropdown_choices, outputs=[tts_ref_audio_project_dropdown]
                    )
                    tts_ref_audio_project_dropdown.change(select_tts_ref_audio_from_project, inputs=[tts_ref_audio_project_dropdown], outputs=[tts_ref_audio]) # connect select function
                    tts_ref_audio_project_refresh_btn.click(update_batch_tts_ref_audio_project_dropdown_choices, outputs=[tts_ref_audio_project_dropdown])


                # --- Tab 1.2: Voice to Voice (Now includes batch sweep options) ---
                with gr.Tab("Voice to Voice", id="vc_sub_tab") as vc_tab:
                    gr.Markdown(
                        """
                        ## Voice to Voice Conversion
                        Convert the timbre of an input audio to match a reference speaker's voice.
                        """
                    )
                    with gr.Row(equal_height=False, variant="panel"):
                        with gr.Column(scale=1): # Left column for inputs
                            gr.Markdown("### Source Audio")
                            vc_input_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload or record the audio you want to process.", interactive=True)
                            with gr.Row(visible=False) as vc_input_audio_project_row: # Hidden by default, controls visibility of its children
                                vc_load_input_audio_from_project_btn = gr.Button("Load Source Audio from Project") # Visible on project active
                                vc_input_audio_project_dropdown = gr.Dropdown(label="Select Source Audio from Project", visible=False, allow_custom_value=False)
                                vc_input_audio_project_refresh_btn = gr.Button("Refresh", visible=False)

                            gr.Markdown("### Reference Voice (Speaker) (Max. 40s)")
                            vc_target_voice = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload or record audio to define the target speaker's voice.", interactive=True)
                            with gr.Row(visible=False) as vc_target_voice_project_row: # Hidden by default, controls visibility of its children
                                vc_load_target_voice_from_project_btn = gr.Button("Load Reference Voice from Project") # Visible on project active
                                vc_target_voice_project_dropdown = gr.Dropdown(label="Select Reference Voice from Project", visible=False, allow_custom_value=False)
                                vc_target_voice_project_refresh_btn = gr.Button("Refresh", visible=False)
                                
                            gr.Markdown("### Core Generation Parameters")
                            with gr.Group(): # Visually group these sliders
                                vc_inference_cfg_rate_slider = gr.Slider(
                                    minimum=0.0,
                                    maximum=30.0,
                                    step=0.1,
                                    value=default_inference_cfg_rate,
                                    label="Inference CFG Rate", 
                                    info="Higher values for stronger speaker timbre similarity",
                                    interactive=True
                                )
                                vc_sigma_min_number = gr.Number(
                                    value=default_sigma_min,
                                    label="Sigma Min", 
                                    info="Diffusion parameter, usually a small positive value",
                                    minimum=1e-07,
                                    maximum=1e-05,
                                    step=1e-07,
                                    interactive=True
                                )
                                
                            # Batch Configuration Section (collapsible)
                            with gr.Accordion("Batch Parameter Sweep Options", open=False) as vc_batch_accordion:
                                vc_batch_mode_checkbox = gr.Checkbox(label="Enable Batch Parameter Sweep", value=False, interactive=True)
                                with gr.Column(visible=False) as vc_batch_options_group:
                                    gr.Markdown("#### Batch Configuration")
                                    vc_batch_parameter_dropdown = gr.Dropdown(
                                        choices=["Inference CFG Rate", "Sigma Min"],
                                        label="Parameter to Vary in Batch", 
                                        value="Inference CFG Rate", 
                                        interactive=True
                                    )
                                    vc_batch_values_textbox = gr.Textbox(
                                        label="Comma-separated Values for Batch Sweep", 
                                        placeholder="e.g., 0.5, 0.7, 0.9 (for CFG Rate) or 1e-6, 5e-6, 1e-5 (for Sigma Min)",
                                        interactive=True
                                    )
                                    
                            # Link checkbox to visibility of batch options group
                            vc_batch_mode_checkbox.change(
                                fn=toggle_vc_batch_options_visibility,
                                inputs=[vc_batch_mode_checkbox],
                                outputs=[vc_batch_options_group]
                            )

                            vc_run_btn = gr.Button("🚀 Start Voice Conversion", variant="primary")

                        with gr.Column(scale=1): # Right column for logs and output
                            gr.Markdown("### Conversion Log")
                            vc_process_log = gr.Textbox(
                                label="Log",
                                lines=15, 
                                interactive=False 
                            ) 
                            gr.Markdown("### Converted Audio Output")
                            # Two output components, their visibility managed by yield_vc_updates
                            vc_audio_output = gr.Audio(label="Playback", type="numpy", visible=False)
                            vc_output_files = gr.File(label="Download Generated Audio(s)", visible=False)

                    # VC Button Click Event
                    vc_run_btn.click(
                        fn=generate_vc,
                        inputs=[
                            vc_input_audio,
                            vc_target_voice,
                            vc_inference_cfg_rate_slider,
                            vc_sigma_min_number,
                            # New batch related inputs
                            vc_batch_mode_checkbox, 
                            vc_batch_parameter_dropdown, 
                            vc_batch_values_textbox, 
                        ],
                        outputs=[
                            vc_process_log,
                            vc_audio_output,
                            vc_output_files 
                        ],
                        show_progress='minimal' 
                    )

                    # VC Project-specific UI Interactions
                    vc_load_input_audio_from_project_btn.click(
                        fn=lambda: gr.update(value=None), outputs=[vc_input_audio]
                    ).then(
                        fn=update_vc_input_audio_project_dropdown_choices, outputs=[vc_input_audio_project_dropdown]
                    )
                    vc_input_audio_project_dropdown.change(select_vc_input_audio_from_project, inputs=[vc_input_audio_project_dropdown], outputs=[vc_input_audio])
                    vc_input_audio_project_refresh_btn.click(update_vc_input_audio_project_dropdown_choices, outputs=[vc_input_audio_project_dropdown])

                    vc_load_target_voice_from_project_btn.click(
                        fn=lambda: gr.update(value=None), outputs=[vc_target_voice]
                    ).then(
                        fn=update_vc_target_voice_project_dropdown_choices, outputs=[vc_target_voice_project_dropdown]
                    )
                    vc_target_voice_project_dropdown.change(select_vc_target_voice_from_project, inputs=[vc_target_voice_project_dropdown], outputs=[vc_target_voice])
                    vc_target_voice_project_refresh_btn.click(update_vc_target_voice_project_dropdown_choices, outputs=[vc_target_voice_project_dropdown])


        # --- Main Tab 2: Batch Generation ---
        with gr.Tab("Batch Generation"):
            gr.Markdown(
                """
                ## Batch Generation: Bulk Processing of Prepared Data
                Process multiple text or audio files using a consistent set of parameters.
                Combined with the Data Preparation tools, this allows for powerful automated workflows.
                """
            )
            with gr.Tabs(): # Nested tabs for Text to Voice Batch and Voice to Voice Batch
                # --- Tab 3.2: Text to Voice Batch ---
                with gr.Tab("Text to Voice Batch"):
                    gr.Markdown(
                        """
                        ### Batch Text to Voice Synthesis
                        Synthesize speech for a collection of text files using one reference voice.
                        """
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### Input Text Files for Batch")
                            batch_tts_zip_upload = gr.File(
                                label="Upload a .zip file containing .txt chunks", 
                                file_types=[".zip"], 
                                type="filepath",
                                interactive=True
                            )
                            with gr.Row():
                                batch_tts_load_zip_btn = gr.Button("Load Files from .zip")
                                batch_tts_clear_files_btn = gr.Button("Clear Selected Files")
                                
                            with gr.Row(visible=False) as batch_tts_project_files_row: # Hidden by default, for project integration
                                batch_tts_use_project_files_btn = gr.Button("Use Files from Project's Processed Text Folder")
                                batch_tts_project_files_refresh_btn = gr.Button("Refresh Project Files", visible=False)

                            batch_tts_files_display = gr.Textbox( 
                                label="Files selected for processing:", 
                                value="None selected.", 
                                lines=5, 
                                interactive=False, 
                                elem_id="batch_tts_files_display" 
                            )
                            # Hidden state to store actual file paths
                            batch_tts_files_to_process_state = gr.State([]) 

                            gr.Markdown("#### Reference Audio (Speaker Voice) for Batch (Max. 40s)")
                            batch_tts_ref_audio_input = gr.Audio(
                                sources=["upload"], # Only upload for batch reference, mic not practical
                                type="filepath", 
                                label="Upload audio to define the target speaker's voice for ALL batch generations.", 
                                interactive=True
                            )
                            with gr.Row(visible=False) as batch_tts_ref_audio_project_row:
                                batch_tts_load_ref_audio_from_project_btn = gr.Button("Load Ref Audio from Project Voice Conversion")
                                batch_tts_ref_audio_dropdown = gr.Dropdown(label="Select Reference Audio from Project", visible=False, allow_custom_value=False)
                                batch_tts_ref_audio_refresh_btn = gr.Button("Refresh", visible=False)

                            gr.Markdown("#### Batch TTS Parameters (Fixed for all files)")
                            batch_tts_exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration", info="Neutral = 0.5, extreme values can be unstable.", value=.5)
                            batch_tts_cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", info="Classifier-Free Guidance weight for pacing and clarity.", value=0.5)
                            batch_tts_seed_num = gr.Number(value=0, label="Random seed (0 for random)", info="Set a seed for reproducible generations (0 for truly random).")
                            batch_tts_temp = gr.Slider(0.05, 5, step=.05, label="Temperature", info="Controls the randomness of the output speech, higher values for more variability.", value=.8)

                            batch_tts_concatenate_checkbox = gr.Checkbox(label="Concatenate all generated audio into one file", value=False)
                            batch_tts_run_btn = gr.Button("🚀 Run Batch Text to Voice", variant="primary")

                        with gr.Column(scale=1):
                            gr.Markdown("#### Batch TTS Log")
                            batch_tts_log = gr.Textbox(label="Log", lines=10, interactive=False) 
                            gr.Markdown("#### Batch Generated Audio Output")
                            batch_tts_output_files = gr.File(label="Download Generated Audio(s)", file_count="multiple", visible=False)
                            
                    # --- Batch TTS Event Handling ---
                    batch_tts_load_zip_btn.click(
                        fn=handle_batch_tts_zip_upload,
                        inputs=[batch_tts_zip_upload],
                        outputs=[batch_tts_files_display, batch_tts_files_to_process_state, batch_tts_log]
                    )
                    
                    batch_tts_use_project_files_btn.click(
                        fn=load_batch_tts_project_files,
                        inputs=[],
                        outputs=[batch_tts_files_display, batch_tts_files_to_process_state, batch_tts_log]
                    )

                    batch_tts_clear_files_btn.click(
                        fn=clear_batch_tts_files,
                        inputs=[],
                        outputs=[batch_tts_files_display, batch_tts_files_to_process_state, batch_tts_log]
                    )

                    batch_tts_load_ref_audio_from_project_btn.click(lambda: gr.update(visible=True), outputs=[batch_tts_ref_audio_dropdown])
                    batch_tts_load_ref_audio_from_project_btn.click(update_batch_tts_ref_audio_project_dropdown_choices, outputs=[batch_tts_ref_audio_dropdown])
                    batch_tts_load_ref_audio_from_project_btn.click(lambda: gr.update(visible=True), outputs=[batch_tts_ref_audio_refresh_btn])
                    batch_tts_ref_audio_dropdown.change(select_batch_tts_ref_audio_from_project, inputs=[batch_tts_ref_audio_dropdown], outputs=[batch_tts_ref_audio_input])
                    batch_tts_ref_audio_refresh_btn.click(update_batch_tts_ref_audio_project_dropdown_choices, outputs=[batch_tts_ref_audio_dropdown])

                    batch_tts_run_btn.click(
                        fn=run_batch_tts,
                        inputs=[
                            batch_tts_ref_audio_input,
                            batch_tts_exaggeration,
                            batch_tts_temp,
                            batch_tts_seed_num,
                            batch_tts_cfg_weight,
                            batch_tts_files_to_process_state,
                            batch_tts_concatenate_checkbox
                        ],
                        outputs=[
                            batch_tts_log,
                            batch_tts_output_files
                        ],
                        show_progress='minimal'
                    )


                # --- Tab 3.3: Voice to Voice Batch ---
                with gr.Tab("Voice to Voice Batch"):
                    gr.Markdown(
                        """
                        ### Batch Voice to Voice Conversion
                        Convert the voice timbre of a collection of audio files using one reference voice.
                        """
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### Input Audio Files for Batch")
                            batch_vc_zip_upload = gr.File(
                                label="Upload a .zip file containing audio chunks (.wav, .mp3, .flac)", 
                                file_types=[".zip"], 
                                type="filepath",
                                interactive=True
                            )
                            with gr.Row():
                                batch_vc_load_zip_btn = gr.Button("Load Files from .zip")
                                batch_vc_clear_files_btn = gr.Button("Clear Selected Files")
                                
                            with gr.Row(visible=False) as batch_vc_project_files_row:
                                batch_vc_use_project_files_btn = gr.Button("Use Files from Project's Processed Audio Folder")
                                batch_vc_project_files_refresh_btn = gr.Button("Refresh Project Files", visible=False)

                            batch_vc_files_display = gr.Textbox(
                                label="Files selected for processing:", 
                                value="None selected.", 
                                lines=5, 
                                interactive=False
                            )
                            batch_vc_files_to_process_state = gr.State([])

                            gr.Markdown("#### Reference Voice (Target Speaker) for Batch (Max. 40s)")
                            batch_vc_ref_voice_input = gr.Audio(
                                sources=["upload"], 
                                type="filepath", 
                                label="Upload audio for the target speaker's voice for ALL batch conversions.", 
                                interactive=True
                            )
                            with gr.Row(visible=False) as batch_vc_ref_voice_project_row:
                                batch_vc_load_ref_voice_from_project_btn = gr.Button("Load Ref Voice from Project Voice Conversion")
                                batch_vc_ref_voice_dropdown = gr.Dropdown(label="Select Reference Voice from Project", visible=False, allow_custom_value=False)
                                batch_vc_ref_voice_refresh_btn = gr.Button("Refresh", visible=False)

                            gr.Markdown("#### Batch VC Parameters (Fixed for all files)")
                            batch_vc_inference_cfg_rate_slider = gr.Slider(
                                minimum=0.0, maximum=30.0, step=0.1, value=default_inference_cfg_rate,
                                label="Inference CFG Rate", info="Higher values for stronger speaker timbre similarity", interactive=True
                            )
                            batch_vc_sigma_min_number = gr.Number(
                                value=default_sigma_min, label="Sigma Min", info="Diffusion parameter, usually a small positive value",
                                minimum=1e-07, maximum=1e-05, step=1e-07, interactive=True
                            )

                            batch_vc_concatenate_checkbox = gr.Checkbox(label="Concatenate all converted audio into one file", value=False)
                            batch_vc_run_btn = gr.Button("🚀 Run Batch Voice Conversion", variant="primary")

                        with gr.Column(scale=1):
                            gr.Markdown("#### Batch VC Log")
                            batch_vc_log = gr.Textbox(label="Log", lines=10, interactive=False)
                            gr.Markdown("#### Batch Converted Audio Output")
                            batch_vc_output_files = gr.File(label="Download Converted Audio(s)", file_count="multiple", visible=False)

                    # --- Batch VC Event Handling ---
                    batch_vc_load_zip_btn.click(
                        fn=handle_batch_vc_zip_upload,
                        inputs=[batch_vc_zip_upload],
                        outputs=[batch_vc_files_display, batch_vc_files_to_process_state, batch_vc_log]
                    )
                    batch_vc_use_project_files_btn.click(
                        fn=load_batch_vc_project_files,
                        inputs=[],
                        outputs=[batch_vc_files_display, batch_vc_files_to_process_state, batch_vc_log]
                    )
                    batch_vc_clear_files_btn.click(
                        fn=clear_batch_vc_files,
                        inputs=[],
                        outputs=[batch_vc_files_display, batch_vc_files_to_process_state, batch_vc_log]
                    )
                    batch_vc_load_ref_voice_from_project_btn.click(lambda: gr.update(visible=True), outputs=[batch_vc_ref_voice_dropdown])
                    batch_vc_load_ref_voice_from_project_btn.click(update_batch_vc_ref_voice_project_dropdown_choices, outputs=[batch_vc_ref_voice_dropdown])
                    batch_vc_load_ref_voice_from_project_btn.click(lambda: gr.update(visible=True), outputs=[batch_vc_ref_voice_refresh_btn])
                    batch_vc_ref_voice_dropdown.change(select_batch_vc_ref_voice_from_project, inputs=[batch_vc_ref_voice_dropdown], outputs=[batch_vc_ref_voice_input])
                    batch_vc_ref_voice_refresh_btn.click(update_batch_vc_ref_voice_project_dropdown_choices, outputs=[batch_vc_ref_voice_dropdown])
                    
                    batch_vc_run_btn.click(
                        fn=run_batch_vc,
                        inputs=[
                            batch_vc_ref_voice_input,
                            batch_vc_inference_cfg_rate_slider,
                            batch_vc_sigma_min_number,
                            batch_vc_files_to_process_state,
                            batch_vc_concatenate_checkbox
                        ],
                        outputs=[
                            batch_vc_log,
                            batch_vc_output_files
                        ],
                        show_progress='minimal'
                    )
        
        # --- Data Preparation ---
        with gr.Tab("Data Preparation") as data_prep_tab:
            gr.Markdown(
                """
                ## Data Preparation Tools
                Tools to help you prepare your raw audio and text data into model-ready chunks.
                """
            )
            with gr.Tabs(): # Nested Tabs for Data Preparation modes
                # --- Tab 3.1: Text Splitting ---
                with gr.Tab("Text Splitting"):
                    gr.Markdown(
                        """
                        ### Text Document Splitter
                        Split long text files into smaller, manageable chunks based on sentence boundaries and a maximum character limit.
                        Outputs are saved to your active project's `processed_text/` directory.
                        """
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### Input Text File")
                            dp_text_input_file = gr.File(
                                label="Upload a .txt file for splitting", 
                                type="filepath", # Corrected type: 'filepath' is string literal
                                file_types=[".txt"],
                                interactive=True
                            )
                            with gr.Row(visible=False) as dp_text_project_row: # Hidden by default, for project integration
                                dp_load_text_from_project_btn = gr.Button("Load .txt from Project's Input Files")
                                dp_text_project_dropdown = gr.Dropdown(label="Select .txt from Project Input Files", visible=False, allow_custom_value=False)
                                dp_text_project_refresh_btn = gr.Button("Refresh", visible=False)

                            gr.Markdown("#### Splitting Parameters")
                            dp_max_chars_per_chunk = gr.Slider(
                                minimum=100,
                                maximum=5000,
                                step=100,
                                value=300,
                                label="Max Characters Per Chunk",
                                info="Each generated chunk will have a maximum of approx. this many characters, respecting sentence boundaries."
                            )
                            dp_split_text_btn = gr.Button("✂️ Split Text File", variant="primary")
                            
                        with gr.Column(scale=1):
                            gr.Markdown("#### Text Splitting Log")
                            dp_text_splitting_log = gr.Textbox(label="Log", lines=10, interactive=False)
                            gr.Markdown("#### Generated Text Chunks")
                            dp_generated_text_chunks = gr.File(label="Download Generated Text Chunks", file_count="multiple", visible=False)
                            
                    # Text Splitting Button Click Event
                    dp_split_text_btn.click(
                        fn=split_text_document,
                        inputs=[
                            dp_text_input_file,
                            dp_max_chars_per_chunk
                        ],
                        outputs=[
                            dp_text_splitting_log,
                            dp_generated_text_chunks
                        ],
                        show_progress='minimal'
                    )

                    # Text Splitting Project-specific UI Interactions
                    dp_load_text_from_project_btn.click(
                        fn=lambda: gr.update(value=None), outputs=[dp_text_input_file]
                    ).then(
                        fn=update_dp_text_project_dropdown_choices, outputs=[dp_text_project_dropdown]
                    )
                    dp_text_project_dropdown.change(select_dp_text_from_project, inputs=[dp_text_project_dropdown], outputs=[dp_text_input_file])
                    dp_text_project_refresh_btn.click(update_dp_text_project_dropdown_choices, outputs=[dp_text_project_dropdown])

                # --- Tab 3.2: Audio Splitting ---
                with gr.Tab("Audio Splitting"):
                    gr.Markdown(
                        """
                        ### Audio Document Splitter
                        Split long audio files into smaller, manageable chunks, prioritizing cuts in silence.
                        Outputs are saved to your active project's `processed_audio/` directory.
                        """
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### Input Audio File")
                            dp_audio_input_file = gr.Audio(
                                label="Upload an audio file (.wav, .mp3, .flac) for splitting",
                                type="filepath", 
                                sources=["upload"], # No microphone here, assumes longer files
                                interactive=True
                            )
                            with gr.Row(visible=False) as dp_audio_project_row:
                                dp_load_audio_from_project_btn = gr.Button("Load Audio from Project's Input Files")
                                dp_audio_project_dropdown = gr.Dropdown(label="Select Audio from Project Input Files", visible=False, allow_custom_value=False)
                                dp_audio_project_refresh_btn = gr.Button("Refresh", visible=False)

                            gr.Markdown("#### Splitting Parameters")
                            dp_max_audio_duration_sec = gr.Slider(
                                minimum=10,
                                maximum=120, # Allow up to 2 minutes per chunk
                                step=1,
                                value=39,
                                label="Max Duration Per Chunk (seconds)",
                                info="Each generated audio chunk will aim for this maximum duration, splitting at silence points."
                            )
                            dp_min_audio_duration_sec = gr.Slider(
                                minimum=1, # Minimum 1 second
                                maximum=10, # Up to 10 seconds as a minimum
                                step=0.5,
                                value=10, 
                                label="Min Duration Per Chunk (seconds)",
                                info="Chunks shorter than this will be merged with adjacent segments to meet the minimum."
                            )
                            dp_min_silence_for_cut_ms = gr.Slider(
                                minimum=50,
                                maximum=1000, # Up to 1 second
                                step=50,
                                value=200, 
                                label="Minimum Silence for Cut Detection (ms)",
                                info="How long a silence must be to be considered a good cut point."
                            )
                            dp_split_audio_btn = gr.Button("✂️ Split Audio File", variant="primary")
                            
                        with gr.Column(scale=1):
                            gr.Markdown("#### Audio Splitting Log")
                            dp_audio_splitting_log = gr.Textbox(label="Log", lines=10, interactive=False)
                            gr.Markdown("#### Generated Audio Chunks")
                            dp_generated_audio_chunks = gr.File(label="Download Generated Audio Chunks", file_count="multiple", visible=False)

                    # Audio Splitting Button Click Event
                    dp_split_audio_btn.click(
                        fn=split_audio_document,
                        inputs=[
                            dp_audio_input_file,
                            dp_max_audio_duration_sec,
                            dp_min_audio_duration_sec,
                            dp_min_silence_for_cut_ms 
                        ],
                        outputs=[
                            dp_audio_splitting_log,
                            dp_generated_audio_chunks
                        ],
                        show_progress='minimal'
                    )

                    # Audio Splitting Project-specific UI Interactions
                    dp_load_audio_from_project_btn.click(
                        fn=lambda: gr.update(value=None), outputs=[dp_audio_input_file]
                    ).then(
                        fn=update_dp_audio_project_dropdown_choices, outputs=[dp_audio_project_dropdown]
                    )
                    dp_audio_project_dropdown.change(select_dp_audio_from_project, inputs=[dp_audio_project_dropdown], outputs=[dp_audio_input_file])
                    dp_audio_project_refresh_btn.click(update_dp_audio_project_dropdown_choices, outputs=[dp_audio_project_dropdown])
                    
                # --- NEW: Edit Project Data Tab ---
                with gr.Tab("Edit Project Data", visible=False) as edit_project_data_tab:
                    gr.Markdown(
                        """
                        ## Edit Project Data: Fine-tune Your Prepared Files
                        View and directly edit text files, or regenerate specific audio chunks.
                        """
                    )
                    with gr.Tabs():
                        # --- Edit Text Files Sub-tab ---
                        with gr.Tab("Edit Text Files"):
                            gr.Markdown(
                                """
                                ### Edit Text Files in Processed Text Folder
                                Select a text file, edit its content, and save the changes directly.
                                """
                            )
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Markdown("#### Select File to Edit")
                                    edit_text_file_dropdown = gr.Dropdown(
                                        label="Select .txt file from processed_text/",
                                        interactive=True,
                                        value=None # Starts with no selection
                                    )
                                    with gr.Row(): # New row for buttons below dropdown
                                        edit_text_refresh_btn = gr.Button("Refresh File List") 
                                        edit_text_random_btn = gr.Button("Load Random File") # <--- NEW BUTTON

                                    gr.Markdown("#### Edit Content")
                                    edit_text_content = gr.Textbox(
                                        label="File Content",
                                        lines=10,
                                        interactive=False, 
                                        value="" 
                                    )
                                    save_edited_text_btn = gr.Button("Save Changes", variant="primary", interactive=False) 

                                with gr.Column(scale=1): 
                                    
                                    gr.Markdown("#### Status / Log")
                                    edit_text_log = gr.Textbox(
                                        label="Log and Status Messages",
                                        lines=15,
                                        interactive=False,
                                        value="Select a project and then a text file to begin editing." 
                                    )
                                    with gr.Row(variant="compact"): 
                                        gr.Markdown("#### Navigation", elem_classes="text-center") 
                                        edit_text_prev_btn = gr.Button("⬆️ Prev", size="sm", interactive=False) 
                                        edit_text_next_btn = gr.Button("⬇️ Next", size="sm", interactive=False)
                                    current_editable_files_state = gr.State([])

                            # Event Handlers for Edit Text Files
                            edit_text_refresh_btn.click(
                                fn=update_edit_text_dropdown_choices,
                                outputs=[edit_text_file_dropdown, current_editable_files_state] 
                            )
                            
                            # *** IMPORTANT FIX HERE ***: Corrected outputs to match load_edit_text_content_only's extended returns
                            edit_text_file_dropdown.change(
                                fn=load_edit_text_content_only, 
                                inputs=[edit_text_file_dropdown],
                                outputs=[edit_text_content, save_edited_text_btn, edit_text_prev_btn, edit_text_next_btn, edit_text_log, current_editable_files_state] # CORRECTED LINE
                            )

                            save_edited_text_btn.click(
                                fn=save_edited_text_content,
                                inputs=[edit_text_file_dropdown, edit_text_content],
                                outputs=[edit_text_content, save_edited_text_btn, edit_text_log] 
                            )
                            
                            edit_text_prev_btn.click(
                                fn=navigate_text_file,
                                inputs=[edit_text_file_dropdown, current_editable_files_state, gr.State(-1)], 
                                outputs=[edit_text_file_dropdown, edit_text_log, edit_text_prev_btn, edit_text_next_btn] 
                            )

                            edit_text_next_btn.click(
                                fn=navigate_text_file,
                                inputs=[edit_text_file_dropdown, current_editable_files_state, gr.State(1)], 
                                outputs=[edit_text_file_dropdown, edit_text_log, edit_text_prev_btn, edit_text_next_btn] 
                            )

                            # <--- NEW: Connect Load Random File button
                            edit_text_random_btn.click(
                                fn=load_random_text_file,
                                inputs=[],
                                outputs=[edit_text_file_dropdown, edit_text_content, edit_text_prev_btn, edit_text_next_btn, edit_text_log, current_editable_files_state]
                            )


                        # --- NEW: Regenerate Audio Sub-tab ---
                        with gr.Tab("Regenerate Project Audio", visible=True) as regenerate_audio_sub_tab:
                            gr.Markdown(
                                """
                                ### Regenerate Audio from Batch TTS
                                Review audio files from a batch TTS run, send them to the single TTS for modification, and replace them.
                                """
                            )
                            # Internal states for this tab
                            _regen_selected_batch_run_path_state = gr.State(None)
                            _regen_selected_audio_file_path_to_replace_state = gr.State(None)
                            _regen_selected_audio_original_txt_path_state = gr.State(None)
                            _regen_batch_ref_audio_path_state = gr.State(None)
                            _last_single_tts_output_path_gr_state = gr.State(None) # To securely pass the path

                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Markdown("#### 1. Select Batch TTS Run")
                                    regen_select_batch_run_dropdown = gr.Dropdown(label="Select Batch TTS Run (Newest First)", choices=[])
                                    regen_refresh_batch_runs_btn = gr.Button("Refresh Batch Runs List")
                                    
                                    gr.Markdown("#### 2. Select Audio File from Batch")
                                    regen_select_audio_file_dropdown = gr.Dropdown(label="Select Audio File to Regenerate", choices=[], interactive=False)
                                    regen_audio_preview = gr.Audio(label="Preview Selected Audio", type="filepath", interactive=False)

                                with gr.Column(scale=1):
                                    gr.Markdown("#### 3. Actions")
                                    regen_send_to_tts_btn = gr.Button("➡️ Send to Single Text to Voice for Editing", interactive=False)
                                    regen_replace_with_single_gen_btn = gr.Button("💾 Replace Selected Batch Audio with Last Single TTS Generation", variant="primary", interactive=False)
                                    regen_concatenate_btn = gr.Button("🎶 Concatenate All Audios in this Batch Run", variant="secondary", interactive=False)
                                    
                                    gr.Markdown("---")
                                    regen_status_log = gr.Textbox(label="Regeneration Status Log", lines=10, interactive=False,
                                                                    value="Load a project, then select a batch run and an audio file.")
                                    regen_concatenated_output_file = gr.File(label="Download Concatenated Audio", visible=False)

                            # Event Handlers for Regenerate Audio
                            regen_refresh_batch_runs_btn.click(
                                fn=lambda: gr.update(choices=list_batch_tts_runs()),
                                outputs=[regen_select_batch_run_dropdown]
                            )
                            
                            regen_select_batch_run_dropdown.change(
                                fn=load_batch_tts_run_files,
                                inputs=[regen_select_batch_run_dropdown],
                                outputs=[regen_select_audio_file_dropdown, regen_status_log, _regen_selected_batch_run_path_state, _regen_batch_ref_audio_path_state, regen_send_to_tts_btn, regen_concatenate_btn, regen_concatenated_output_file]
                            )
                            
                            regen_select_audio_file_dropdown.change(
                                fn=load_selected_batch_audio_for_preview,
                                inputs=[regen_select_audio_file_dropdown, _regen_selected_batch_run_path_state],
                                outputs=[regen_audio_preview, regen_status_log, _regen_selected_audio_file_path_to_replace_state, _regen_selected_audio_original_txt_path_state, regen_send_to_tts_btn]
                            )

                            regen_send_to_tts_btn.click(
                                fn=action_send_to_single_tts,
                                inputs=[_regen_selected_audio_original_txt_path_state, _regen_batch_ref_audio_path_state],
                                outputs=[main_tabs, single_generation_sub_tabs, tts_text, tts_ref_audio, regen_status_log, regen_replace_with_single_gen_btn],
                                show_progress="minimal"
                            )
                            
                            def get_last_single_tts_path_for_gradio(): # Wrapper to access Python global
                                return _last_single_tts_output_path_state_value

                            regen_replace_with_single_gen_btn.click(
                                fn=get_last_single_tts_path_for_gradio, # First, get the path into a Gradio state
                                outputs=[_last_single_tts_output_path_gr_state]
                            ).then(
                                fn=action_replace_batch_audio, # Then, use that state as input
                                inputs=[_regen_selected_audio_file_path_to_replace_state, _last_single_tts_output_path_gr_state],
                                outputs=[regen_audio_preview, regen_status_log, regen_replace_with_single_gen_btn, _last_single_tts_output_path_gr_state]
                            )

                            regen_concatenate_btn.click(
                                fn=concatenate_batch_run_audios,
                                inputs=[_regen_selected_batch_run_path_state],
                                outputs=[regen_status_log, regen_concatenated_output_file],
                                show_progress="minimal"
                            )


        # --- PROJECTS Tab ---
        with gr.Tab("Projects"):
            gr.Markdown(
                """
                ## Project Management: Organize Your Work 📂
                Create, select, and manage dedicated workspaces for your audio generation tasks.
                All generated audios will be saved to the currently active project folder.
                """
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.Markdown("### Create New Project")
                    new_project_name_input = gr.Textbox(label="New Project Name", placeholder="Enter a unique name for your project (e.g., MyVoiceExperiment)")
                    create_project_btn = gr.Button("➕ Create & Select New Project", variant="primary")
                    
                    gr.Markdown("### Select Existing Project")
                    
                    _initial_projects_list = list_projects()
                    
                    project_dropdown = gr.Dropdown(
                        label="Select Current Project", 
                        choices=_initial_projects_list, 
                        value=None, 
                        interactive=True,
                        allow_custom_value=False
                    )
                    current_project_path_display = gr.Textbox(
                        label="Current Project Root Path", 
                        value="", 
                        interactive=False
                    )
                    project_status_message = gr.Textbox(label="Project Status", interactive=False)
                    
                    unload_project_btn = gr.Button("↩️ Unload Current Project", variant="secondary")

                    gr.Markdown("### Project Utilities")
                    project_upload_target_folder = gr.Dropdown(
                        label="Upload Files to Subfolder",
                        choices=[("Input Files (general uploads)", "input_files"), ("Voice Conversion (reference voices)", "voice_conversion")],
                        value="input_files", 
                        interactive=True,
                        allow_custom_value=False 
                    )
                    project_file_upload = gr.File(
                        label="Select Files for Upload", 
                        file_count="multiple", 
                        type="filepath", 
                        file_types=["audio", ".txt", ".mp3", ".wav", ".flac"], 
                    )
                    upload_to_project_btn = gr.Button("📤 Upload Selected Files")
                    upload_status_message = gr.Textbox(label="Upload Status", interactive=False)
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Project Contents")
                    gr.Markdown(
                        """
                        This section displays the file structure within your active project.
                        You can select and delete files/folders directly here.
                        **Caution: Deletion is permanent and has no confirmation.**
                        """
                    )
                    
                    # File Explorer Section
                    project_file_explorer = gr.FileExplorer(
                        root_dir="",
                        file_count="multiple",
                        height=400,
                        label="Explore Current Project Files",
                        interactive=False 
                    )
                    with gr.Row():
                        refresh_project_view_btn = gr.Button("🔄 Refresh View", interactive=False)
                        download_selected_project_files_btn = gr.Button("⬇️ Download Selected", interactive=False)
                        delete_selected_project_files_btn = gr.Button("🗑️ Delete Selected", interactive=False, variant="secondary")
                    
                    project_download_output = gr.File(label="Download Zip", visible=False)
                    delete_status_message = gr.Textbox(label="File Operations Log", interactive=False, lines=3)

            all_project_dependent_ui_elements = [ # Combined list for easier management
                project_dropdown, 
                current_project_path_display, 
                project_status_message,
                # Feature-specific UI dependent on project
                tts_text_project_row, 
                tts_load_text_from_project_btn,
                tts_text_project_dropdown,
                tts_text_project_refresh_btn,
                tts_ref_audio_project_row, 
                tts_load_ref_audio_from_project_btn,
                tts_ref_audio_project_dropdown,
                tts_ref_audio_project_refresh_btn,
                vc_input_audio_project_row, 
                vc_load_input_audio_from_project_btn,
                vc_input_audio_project_dropdown,
                vc_input_audio_project_refresh_btn,
                vc_target_voice_project_row, 
                vc_load_target_voice_from_project_btn,
                vc_target_voice_project_dropdown,
                vc_target_voice_project_refresh_btn,
                dp_text_project_row, 
                dp_load_text_from_project_btn,
                dp_text_project_dropdown,
                dp_text_project_refresh_btn,
                dp_audio_project_row,
                dp_load_audio_from_project_btn,
                dp_audio_project_dropdown,
                dp_audio_project_refresh_btn,
                batch_tts_project_files_row, 
                batch_tts_use_project_files_btn,
                batch_tts_project_files_refresh_btn,
                batch_tts_ref_audio_project_row, 
                batch_tts_load_ref_audio_from_project_btn,
                batch_tts_ref_audio_dropdown,
                batch_tts_ref_audio_refresh_btn,
                batch_vc_project_files_row,
                batch_vc_use_project_files_btn,
                batch_vc_project_files_refresh_btn,
                batch_vc_ref_voice_project_row,
                batch_vc_load_ref_voice_from_project_btn,
                batch_vc_ref_voice_dropdown,
                batch_vc_ref_voice_refresh_btn,
                # Edit Text Tab (within Edit Project Data)
                edit_project_data_tab, 
                edit_text_file_dropdown, 
                edit_text_content, 
                save_edited_text_btn,
                edit_text_log,
                edit_text_prev_btn,
                edit_text_next_btn,
                current_editable_files_state,
                # New Random button for Edit Text
                edit_text_random_btn,
                # Regenerate Audio Tab (within Edit Project Data)
                regenerate_audio_sub_tab,
                regen_select_batch_run_dropdown,
                regen_refresh_batch_runs_btn,
                regen_select_audio_file_dropdown,
                regen_audio_preview,
                regen_send_to_tts_btn,
                regen_replace_with_single_gen_btn,
                regen_concatenate_btn,
                regen_concatenated_output_file,
                regen_status_log,
                _regen_selected_batch_run_path_state,
                _regen_selected_audio_file_path_to_replace_state,
                _regen_selected_audio_original_txt_path_state,
                _regen_batch_ref_audio_path_state,
                _last_single_tts_output_path_gr_state,
                # NEW Project Browser Elements
                project_file_explorer,
                refresh_project_view_btn,
                download_selected_project_files_btn,
                delete_selected_project_files_btn,
                project_download_output,
                delete_status_message,
            ]


            def _run_project_dependent_ui_updates():
                is_project_active = (_current_project_root_dir != "")
                # This list should correspond to all_project_dependent_ui_elements AFTER the first 3 (dropdown, path, status)
                updates_for_features = [] 

                # Updates for Single Generation TTS tab
                updates_for_features.append(gr.update(visible=is_project_active)) # tts_text_project_row
                updates_for_features.append(gr.update(visible=is_project_active)) # tts_load_text_from_project_btn
                updates_for_features.append(gr.update(visible=is_project_active, choices=list_project_files('processed_text', '.txt') if is_project_active else [], value=None)) 
                updates_for_features.append(gr.update(visible=is_project_active)) # tts_text_project_refresh_btn

                updates_for_features.append(gr.update(visible=is_project_active)) # tts_ref_audio_project_row
                updates_for_features.append(gr.update(visible=is_project_active)) # tts_load_ref_audio_from_project_btn
                updates_for_features.append(gr.update(visible=is_project_active, choices=list_project_files('voice_conversion', '.wav') if is_project_active else [], value=None)) 
                updates_for_features.append(gr.update(visible=is_project_active)) # tts_ref_audio_project_refresh_btn

                # Updates for Single Generation VC tab
                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(visible=is_project_active, choices=list_project_files('processed_audio', '.wav') if is_project_active else [], value=None)) 
                updates_for_features.append(gr.update(visible=is_project_active)) 

                updates_for_features.append(gr.update(visible=is_project_active))
                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(visible=is_project_active, choices=list_project_files('voice_conversion', '.wav') if is_project_active else [], value=None)) 
                updates_for_features.append(gr.update(visible=is_project_active)) 

                # Updates for Data Preparation Text Splitting tab
                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(visible=is_project_active, choices=list_project_files('input_files', '.txt') if is_project_active else [], value=None)) #
                updates_for_features.append(gr.update(visible=is_project_active)) 

                # Updates for Data Preparation Audio Splitting tab
                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(visible=is_project_active, choices=list_project_files('input_files', ['.wav', '.mp3', '.flac']) if is_project_active else [], value=None)) 
                updates_for_features.append(gr.update(visible=is_project_active)) 

                # Updates for Batch Generation TTS tab
                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(visible=is_project_active)) 

                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(visible=is_project_active, choices=list_project_files('voice_conversion', '.wav') if is_project_active else [], value=None)) 
                updates_for_features.append(gr.update(visible=is_project_active)) 

                # Updates for Batch Generation VC tab
                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(visible=is_project_active)) 

                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(visible=is_project_active, choices=list_project_files('voice_conversion', '.wav') if is_project_active else [], value=None)) 
                updates_for_features.append(gr.update(visible=is_project_active)) 

                # Updates for Edit Project Data tab
                processed_text_files_count = len(list_project_files('processed_text', '.txt')) if is_project_active else 0

                updates_for_features.append(gr.update(visible=is_project_active)) # edit_project_data_tab 
                updates_for_features.append(gr.update(choices=list_project_files('processed_text', '.txt') if is_project_active else [], value=None, visible=is_project_active)) 
                updates_for_features.append(gr.update(value="", interactive=False)) 
                updates_for_features.append(gr.update(interactive=False)) 
                updates_for_features.append(gr.update(value="Select a project and then a text file to begin editing." if is_project_active else "No project active.")) 
                updates_for_features.append(gr.update(interactive=False)) 
                updates_for_features.append(gr.update(interactive=False)) 
                updates_for_features.append(gr.update(value=[] if is_project_active else None)) 
                updates_for_features.append(gr.update(interactive=is_project_active and processed_text_files_count > 0)) # Random button interactivity

                # Updates for Regenerate Audio Sub-Tab (within Edit Project Data)
                updates_for_features.append(gr.update(visible=is_project_active)) 
                updates_for_features.append(gr.update(choices=list_batch_tts_runs() if is_project_active else [], value=None, interactive=is_project_active))
                updates_for_features.append(gr.update(interactive=is_project_active)) 
                updates_for_features.append(gr.update(choices=[], value=None, interactive=False)) # Set interactive to False initially
                updates_for_features.append(gr.update(value=None, interactive=False)) 
                updates_for_features.append(gr.update(interactive=False)) # regen_send_to_tts_btn initial state
                updates_for_features.append(gr.update(interactive=False)) 
                updates_for_features.append(gr.update(interactive=False)) # regen_concatenate_btn
                updates_for_features.append(gr.update(visible=False, value=None)) # regen_concatenated_output_file
                updates_for_features.append(gr.update(value="Load a project to see batch runs." if not is_project_active else "Select a batch run."))
                updates_for_features.append(None) 
                updates_for_features.append(None) 
                updates_for_features.append(None) 
                updates_for_features.append(None) 
                updates_for_features.append(None) 

                # UPDATES FOR NEW PROJECT BROWSER ELEMENTS
                updates_for_features.append(gr.update(root_dir=(os.path.abspath(_current_project_root_dir) if is_project_active else ""), interactive=is_project_active, value=[])) # Clear selection
                updates_for_features.append(gr.update(interactive=is_project_active)) # refresh button
                updates_for_features.append(gr.update(interactive=is_project_active)) # download button
                updates_for_features.append(gr.update(interactive=is_project_active)) # delete button
                updates_for_features.append(gr.update(visible=False, value=None)) # download output file
                updates_for_features.append(gr.update(value="" if is_project_active else "Select a project to browse files.", visible=True))


                return updates_for_features

            _trigger_ui_update_on_project_state_change = _run_project_dependent_ui_updates

            # Create project and select project handlers
            create_project_btn.click(
                fn=create_project, 
                inputs=[new_project_name_input], 
                outputs=all_project_dependent_ui_elements 
            )
            project_dropdown.select(
                fn=set_current_project_instance, 
                inputs=[project_dropdown], 
                outputs=all_project_dependent_ui_elements
            )
            unload_project_btn.click(
                fn=set_current_project_instance, 
                inputs=[], 
                outputs=all_project_dependent_ui_elements
            )

            upload_to_project_btn.click(fn=upload_files_to_project, inputs=[project_file_upload, project_upload_target_folder],outputs=[upload_status_message])

            # Connect refresh button (File Explorer)
            refresh_project_view_btn.click(
                fn=refresh_project_explorer_root_update,
                inputs=[current_project_path_display],
                outputs=[project_file_explorer]
            )

            # Connect download button (File Explorer)
            download_selected_project_files_btn.click(
                fn=create_zip_from_selection,
                inputs=[project_file_explorer, current_project_path_display],
                outputs=[delete_status_message, project_download_output]
            )

            # Connect delete button (File Explorer)
            delete_selected_project_files_btn.click(
                fn=delete_project_items_backend,
                inputs=[project_file_explorer, current_project_path_display],
                outputs=[delete_status_message, project_file_explorer]
            )


    # On demo load, ensure project dependent UI is correctly initialized (no project active)
    demo.load(
        fn=lambda: set_current_project_instance(None),
        outputs=all_project_dependent_ui_elements
    )


ensure_projects_base_dir() # Ensure base directory is created on startup

if __name__ == "__main__":
    demo.queue().launch()
