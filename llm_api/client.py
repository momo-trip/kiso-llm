import argparse
import atexit
import base64
import copy
from copy import deepcopy
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial, reduce
from pathlib import Path
from threading import Thread, Timer
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union
import glob
import grp
import http.server
import json
import logging
import os
import platform
import pwd
import random
import re
import select
import shutil
import signal
import socketserver
import stat
import string
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from queue import Queue
from typing import Optional
import chardet
import tempfile
import mimetypes
from databricks.sdk import WorkspaceClient

#
import anthropic
from anthropic import AnthropicBedrock, InternalServerError
import chardet
import clang.cindex
from clang.cindex import (
    CompilationDatabase,
    CompilationDatabaseError,
    Config,
    CursorKind,
    Index,
    TokenKind,
    TranslationUnit,
    TypeKind
)
from graphviz import Digraph
import networkx as nx
import numpy as np
import openai
from openai import AzureOpenAI, BadRequestError, OpenAI
from pydantic import BaseModel
import replicate
import requests
import tiktoken


from utils_api import (
    # normal
    read_json,
    write_json,
    read_file,
    write_file,
    delete_file,
    create_file, 
    copy_file,
    append_file,
    create_permissioned_file,
    create_directory,
    delete_directory,
    copy_directory,
    grant_permissions,
    run_script,
    run_cov_script,
    run_branch_cov_script,
    get_coverage,
    get_branch_covered,
    find_compile_commands_json,
    deduplicate_compile_commands,
    count_file_lines,
    get_timestamp,
    write_testcase,
    run_script_pty,

    # translation
    # obtain_metadata
    get_lined_code,
    get_specific_lined_code,
    get_unit_code,
    get_unit_code_with_location,
)


TESTFILE_COUNTER = 0


@dataclass
class LLMInterface:
    project_id: str
    occupy_path: str
    llm_choice: str
    api_key: str
    full_regions: Dict[str, Any]
    llm_model: str
    temperature: float = 0.7
    timeout: int = 300
    # system_prompt: Optional[str] = None

    # Set paths
    azure_endpoint: Optional[str] = None
    history_path: Optional[str] = None
    token_path: Optional[str] = None
    database_dir: Optional[str] = None
    chat_dir: Optional[str] = None
    count_path: Optional[str] = None
    
    # Others
    # verbose: bool = False
    exp_data: Dict[str, Any] = None
    output_max: int = None #4000. # 128000
    context_window: int = None 
    client_id: int = None 



@dataclass
class TransConfig:
    rust_c_path: str
    c_rust_path: str
    raw_dir: str
    rust_output_dir: str
    #llm_choice: str
    llm_interface: LLMInterface
    target_dir: str
    chat_dir: str
    database_dir : str
    time_path: str
    work_dir: str
    token_path: str
    original_target_dir: str
    build_path: str
    rust_build_path: str
    run_test_path: str
    run_all_path: str
    div_meta_dir: str
    meta_dir: str
    dep_json_path: str
    exp_data: Dict[str, Any]
    repair_count: int
    rust_edition: str
    execute_path: str
    explore_time: float
    notes: str
    cov_target : str
    log_dir: str
    max_iterations: int
    #api_key: str

    # Optional fields
    is_program_path: str = None
    target_path: str = None
    rust_path: str = None # We do not need this at the initial build moment
    c_path: str = None
    div_start_line: int = None
    before_count: int = None
    add_prompt: str = None
    progress_queue: Optional[Queue] = None
    original_run_test_path: str = None
    select: bool = False
    test_path: Optional[str] = None
    file_path: Optional[str] = None
    test_id: Optional[str] = None
    function_name: Optional[str] = None
    main_flag: Optional[bool] = None
    
    # Should be cleaned up
    callee_main_path: str = None
    entry: Dict[str, Any] = None
    test_type: str = None
    snap_dir: str = None
    tmp_dir: str = None
    cov_report_path: str = None
    run_gdb_path: str = None
    run_val_path: str = None
    target_function: str = None
    target_uncovered_ratio: float = None
    target_branch: int = None
    target_line: int = None
    target_end_line: int = None
    target_cmd: str = None
    cmd_list: List[str] = None
    cmd_exe: str = None


    # LLM-related
    model: str = None
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 300
    system_prompt: Optional[str] = None
    verbose: bool = False


@dataclass
class CorConfig:
    one_unit: List[str]
    answer_path: str
    modified_lines: Dict[str, Any]
    key_json: Dict[str, Any]
    raw_dir: str
    rust_output_dir: str
    #llm_choice: str
    llm_interface: LLMInterface
    target_dir: str
    chat_dir: str
    database_dir : str
    time_path: str
    work_dir: str
    token_path: str
    original_target_dir: str
    run_test_path: str
    build_path: str
    rust_build_path: str
    run_all_path: str  #run_path: str
    meta_dir: str
    div_meta_dir: str
    dep_json_path: str
    exp_data: Dict[str, Any]
    repair_count: int
    rust_edition: str
    execute_path: str
    explore_time: float
    notes: str
    cov_target : str
    log_dir: str
    max_iterations: int
    repair_max: int
    #api_key: str

    # Optional fields
    label: str = None
    rust_path: str = None # We do not need this at the initial build moment
    c_path: str = None
    div_start_line: int = None
    before_count: int = None
    add_prompt: str = None
    progress_queue: Optional[Queue] = None
    original_run_test_path: str = None
    select: bool = False
    test_path: Optional[str] = None
    file_path: Optional[str] = None
    test_id: Optional[str] = None
    function_name: Optional[str] = None
    main_flag: Optional[bool] = None
    
    # Should be cleaned up
    callee_main_path: str = None
    entry: Dict[str, Any] = None
    test_type: str = None
    snap_dir: str = None
    tmp_dir: str = None
    cov_report_path: str = None
    run_gdb_path: str = None
    run_val_path: str = None
    target_path: str = None
    target_function: str = None
    target_uncovered_ratio: float = None
    target_branch: int = None
    target_line: int = None
    target_end_line: int = None
    target_cmd: str = None
    cmd_list: List[str] = None
    cmd_exe: str = None

    # LLM-related
    model: str = None
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 300
    system_prompt: Optional[str] = None
    verbose: bool = False



@dataclass
class SemConfig:
    mix_io_dir: str
    c_io_dir: str
    rust_io_dir: str
    build_path: str
    rust_build_path: str
    run_test_path: str
    run_all_path: str
    rust_c_path: str
    c_rust_path: str
    target: str
    #modified_lines: Dict[str, Any]
    #key_json: Dict[str, Any]
    raw_dir: str
    #llm_choice: str
    llm_interface: LLMInterface
    target_dir: str
    chat_dir: str
    database_dir : str
    time_path: str
    work_dir: str
    token_path: str
    original_target_dir: str
    meta_dir: str
    dep_json_path: str
    exp_data: Dict[str, Any]
    repair_count: int
    execute_path: str
    explore_time: float
    notes: str
    cov_target : str
    log_dir: str
    max_iterations: int
    #api_key: str

    # Optional fields
    rust_path: str = None # We do not need this at the initial build moment
    c_path: str = None
    div_start_line: int = None
    before_count: int = None
    add_prompt: str = None
    progress_queue: Optional[Queue] = None
    original_run_test_path: str = None
    select: bool = False
    test_path: Optional[str] = None
    file_path: Optional[str] = None
    test_id: Optional[str] = None
    function_name: Optional[str] = None
    main_flag: Optional[bool] = None
    
    # Should be cleaned up
    callee_main_path: str = None
    entry: Dict[str, Any] = None
    test_type: str = None
    snap_dir: str = None
    tmp_dir: str = None
    cov_report_path: str = None
    run_gdb_path: str = None
    run_val_path: str = None
    target_path: str = None
    target_function: str = None
    target_uncovered_ratio: float = None
    target_branch: int = None
    target_line: int = None
    target_end_line: int = None
    target_cmd: str = None
    cmd_list: List[str] = None
    cmd_exe: str = None


    # LLM-related
    model: str = None
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 300
    system_prompt: Optional[str] = None
    verbose: bool = False


def add_line_numbers(input_file):
    if not os.path.isfile(input_file):
        return
    
    if not os.path.exists(input_file):
        return
    
    mime_type, _ = mimetypes.guess_type(input_file)
    if mime_type and not mime_type.startswith('text/'):
        print(f"Skip: {input_file} is not a text file (MIME: {mime_type})")
        return
        
    try:
        # First, read the file in binary mode to detect encoding
        with open(input_file, 'rb') as file:
            raw_content = file.read()
            
        if not raw_content:
            print(f"Skip: {input_file} is empty")
            return
            
        # Simple check to determine if it is a binary file
        # Check the ratio of printable ASCII characters (a feature of text files)
        text_chars = len([byte for byte in raw_content[:1024] if 32 <= byte < 127 or byte in (9, 10, 13)])
        if len(raw_content) > 0 and float(text_chars) / min(len(raw_content), 1024) < 0.3:
            print(f"Skip: {input_file} appears to be a binary file")
            return
            
        # Detect encoding using chardet
        detected = chardet.detect(raw_content)
        encoding = detected['encoding']
            
        # Fallback if encoding detection fails or confidence is low
        if not encoding or detected['confidence'] < 0.7:
            encodings = ['utf-8', 'shift-jis', 'euc-jp', 'iso-2022-jp', 'cp932']
            for enc in encodings:
                try:
                    raw_content.decode(enc)
                    encoding = enc
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"Could not read {input_file} with any supported encoding.")
                return
            
        # Create a temporary file and process
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding=encoding) as temp_file:
            content = raw_content.decode(encoding)
            lines = content.splitlines(True)  # Preserve newline characters
                
            # Add line numbers
            for line_number, line in enumerate(lines, start=1):
                numbered_line = f"Line{line_number:5d}:  {line}"
                temp_file.write(numbered_line)
                
        os.replace(temp_file.name, input_file)
        print(f"Wrote file with line numbers to {input_file}.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass
        raise


def adjust_prompt(prompt):
    flat_prompt = []
    for item in prompt:
        if isinstance(item, list):
            flat_prompt.append(json.dumps(item, ensure_ascii=False)) # Convert the list into a JSON-formatted string
        else:
            flat_prompt.append(str(item))
    
    return flat_prompt



def read_specific_lines(filename, start_line, end_line):
    if end_line is None:
        return ""

    start_line = int(start_line)
    end_line = int(end_line)
    
    encodings = ['utf-8', 'latin-1', 'cp932', 'shift_jis', 'euc-jp', 'iso-2022-jp']
    
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return ""
    
    # Check file type (optional)
    is_binary = False
    try:
        with open(filename, 'rb') as test_file:
            sample = test_file.read(1024)
            is_binary = b'\0' in sample  # If NULL byte exists, treat as binary
    except Exception:
        pass
    
    # Attempt to process as a text file
    for encoding in encodings:
        try:
            with open(filename, 'r', encoding=encoding) as file:
                lines = file.readlines()
                if start_line <= 0 or end_line > len(lines):
                    print("Error: Invalid start or end line number.")
                    return ""
                return ''.join(lines[start_line-1:end_line])
        except UnicodeDecodeError:
            continue  # Try the next encoding
        except Exception as e:
            print(f"Error while reading file with {encoding}: {e}")
            continue
    
    # If all encodings fail, read in binary mode
    try:
        with open(filename, 'rb') as file:
            all_bytes = file.read()
            # Force binary to string conversion and split into lines
            text = all_bytes.decode('utf-8', errors='replace')
            lines = text.splitlines(True)
            
            if start_line <= 0 or end_line > len(lines):
                print("Error: Invalid start or end line number.")
                return ""
            
            return ''.join(lines[start_line-1:end_line])
    except Exception as e:
        print(f"Error processing binary file: {e}")
    
    return ""  # If all methods fail


def write_time(time_path, activity, action, repair_target, timestamp=None):
    formatted_time = None
    # if timestamp is not None:
    #     # formatted_time = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
    #     datetime_obj = datetime.strptime(timestamp, '%Y-%m-%d-%H-%M-%S')
    #     formatted_time = datetime_obj.strftime("%Y%m%d_%H%M%S")
    if timestamp is not None:
        # If timestamp is a floating-point number (UNIX timestamp)
        if isinstance(timestamp, float):
            datetime_obj = datetime.fromtimestamp(timestamp)
            # Normalize timestamp to string format as well
            timestamp = datetime_obj.strftime('%Y-%m-%d-%H-%M-%S')
        else:
            datetime_obj = datetime.strptime(timestamp, '%Y-%m-%d-%H-%M-%S')
        
        # Format both cases into the same format
        formatted_time = datetime_obj.strftime("%Y%m%d_%H%M%S")
    

    with open(time_path, 'a') as f:
        f.write(f"{activity},{repair_target},{action},{timestamp},{formatted_time}\n")


def init_prompt_count(count_path):
    data = {}
    if os.path.exists(count_path):
        data = read_json(count_path)
    else:
        create_file(count_path)

    if data is None:
        data = {}

    data["prompt_id"] = str(0).zfill(4)  #0
    write_json(count_path, data)



def clean_prompt(prompt):
    surrogate_pattern = re.compile(r'[\uD800-\uDFFF]') # Regular expression pattern to detect all surrogate characters
    
    def remove_all_surrogates(text):
        if not isinstance(text, str):
            return text
        return surrogate_pattern.sub('', text)
    
    if isinstance(prompt, str): # Process according to the type of the prompt
        return remove_all_surrogates(prompt)
    elif isinstance(prompt, list):
        return [remove_all_surrogates(item) if isinstance(item, str) else item for item in prompt]
    elif isinstance(prompt, dict):
        return {k: clean_prompt(v) for k, v in prompt.items()}
    else:
        return prompt


def create_prompt_string(prompt_items) -> str:
    # Convert any non-string items (including sets) to strings
    string_items = [str(item) for item in prompt_items]
    return "\n".join(string_items)


def load_prompt_count(logging_path):
    if os.path.exists(logging_path):
        data = read_json(logging_path)
        num = data.get("prompt_id", None)
        if num is None:
            num = 0
    else:
        data = {}
        num = 0

    data["prompt_id"] = str(int(num) + 1).zfill(4)  #num + 1
    write_json(logging_path, data)

    return str(int(num)).zfill(4)  #num



def write_prompt(database_dir, signal, data, chat_dir, count_path):
    if count_path is None:
        return 
        
    if signal == 'user':
        prompt_count = load_prompt_count(count_path)
        timestamp = get_timestamp()
        filename = f"{chat_dir}/chat{prompt_count}_user_{timestamp}.txt"
    elif signal == 'llm':
        prompt_count = load_prompt_count(count_path)
        timestamp = get_timestamp()
        filename = f"{chat_dir}/chat{prompt_count}_llm_{timestamp}.txt"

    elif signal == 'request':
        filename = f"{database_dir}/prompt_request.txt"
    elif signal == 'response':
        filename = f"{database_dir}/prompt_response.txt"
        

    try:
        with open(filename, 'w', encoding='utf-8') as file:
            if isinstance(data, (dict, list)):
                json.dump(data, file, ensure_ascii=False, indent=4)
            else:
                if not isinstance(data, str):
                    data = str(data)
                file.write(data)
        print(f"Data was successfully written to {filename}.")
        
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")
        
def update_token(input_token, output_token, token_path):
    # Read data from token_path
    if os.path.exists(token_path):
        data = read_json(token_path)
        # Compute the next prompt_id based on existing data
        prompt_id = len(data)
    else:
        data = []
        prompt_id = 0

    # prompt_id = read_state(logging_path)
    # prompt_id = prompt_id - 1 # important

    # if os.path.exists(token_path):
    #     data = read_json(token_path)
    # else:
    #     data = []
    
    data.append({
        "prompt_id": prompt_id,
        "input_token": input_token,
        "output_token": output_token,
    })

    write_json(token_path, data)


def calculate_tokens(data, encoder):
    if data is None:
        data = []
    return sum(len(encoder.encode(json.dumps(item, ensure_ascii=False))) for item in data)


def deduplicate_sections(data):
    """
    If there are multiple occurrences of specified sections in the conversation history,
    remove all but the most recent one to reduce token usage.

    ## Note: For C code segments, each turn may involve translating different code blocks
    (e.g., first katajainen.c functions, then deflate.c functions). In such cases,
    removing older entries would lose important context (what the LLM has already translated),
    potentially breaking consistency with the assistant's responses.
    
    On the other hand, sections like "## Directory structure" or
    "## Existing code already in" represent the latest snapshot of the state,
    so it is safe to remove older ones.
    """
    section_headers = [
        # macro/pre_process.py
        "## Response format", 

        # compile.py
        "## Directory structure", 
        #"## C code segment",
        "## Rust source code",
        "## Existing code already in",
        "## Module structure of the Rust program",
        "## FFI boundary functions",
        "## Translation rules",

        "## The original C source code",
        "## The converted Rust source code", 
        "## The original C JSON-formatted metadata",

        # semantics.py
        "## Executed test case",
        #"## Error in",
        "## Standard output of execution",
        "## Execution result",
        "## Response rules",
        "## Response modes",
        "## Execution error",
        "## Response to the previous request",
        "## Read data result",
        #"## Code in workspace",
        "## Code in",
        "## The original C source code",
        "## The converted Rust source code",
    ]

    for header in section_headers:
        last_index = None
        indices = []

        for i, item in enumerate(data):
            content = item.get("content", "")
            if not isinstance(content, str):
                continue
            if header in content:
                indices.append(i)
                last_index = i

        if len(indices) <= 1:
            continue

        result = []
        for i, item in enumerate(data):
            if i in indices and i != last_index:
                content = item["content"]
                escaped_header = re.escape(header)
                pattern = rf"{escaped_header}.*?(?=\n## |\Z)"
                new_content = re.sub(pattern, "", content, flags=re.DOTALL).strip()
                if new_content:
                    result.append({**item, "content": new_content})
            else:
                result.append(item)
        data = result  # Process the next header using the updated data

    return data


def _extract_section(content, header):
    escaped = re.escape(header)
    match = re.search(rf"{escaped}.*?(?=\n## |\Z)", content, re.DOTALL)
    return match.group(0) if match else ""


def deduplicate_c_code_segments(data):
    """
    Remove duplicate C code segments only when the content is identical.
    Different segments (e.g., different functions) are preserved.
    """
    header = "## C code segment"
    
    indices = []
    for i, item in enumerate(data):
        content = item.get("content", "")
        if isinstance(content, str) and header in content:
            indices.append(i)
    
    if len(indices) <= 1:
        return data
    
    last_index = indices[-1]
    last_section = _extract_section(data[last_index]["content"], header)
    
    result = []
    for i, item in enumerate(data):
        if i in indices and i != last_index:
            this_section = _extract_section(item["content"], header)
            if this_section == last_section:
                new_content = item["content"].replace(this_section, "").strip()
                if new_content:
                    result.append({**item, "content": new_content})
                continue
        result.append(item)
    
    return result


def deduplicate_prompt(data):

    data = deduplicate_sections(data)

    data = deduplicate_c_code_segments(data)

    return data



def trim_json_data(llm_choice, llm_model, data, max_tokens): # "gpt-3.5-turbo"
    model="gpt-4"
    if llm_choice == "gpt_azure":
        if llm_model == "gpt-4.1": #gpt_model == "gpt-4.1":
            max_tokens = max_tokens * 100

    elif llm_choice == "gpt_azure_databricks": # should be gpt-5
        max_tokens = max_tokens * 80 # 100 caused an error.

    encoder = tiktoken.encoding_for_model(model)

    data = deduplicate_prompt(data)

    # Calculate the total tokens of the current file
    current_total_tokens = calculate_tokens(data, encoder) #sum(len(encoder.encode(json.dumps(item, ensure_ascii=False))) for item in data)
    print(f"Current total tokens in file: {current_total_tokens}")

    # Separate system messages first (so they don't get removed by trimming)
    system_messages = [item for item in data if item.get("role") == "system"]
    non_system_data = [item for item in data if item.get("role") != "system"]

    system_tokens = calculate_tokens(system_messages, encoder)

    total_tokens = 0
    trimmed_data = []
    test_data = []

    # Process data in reverse order (keep from the last data)
    for item in reversed(non_system_data):
        total_tokens = calculate_tokens(trimmed_data, encoder)
        
        test_data = trimmed_data.copy()
        test_data.append(item)
        test_tokens = calculate_tokens(test_data, encoder) + system_tokens
        #print(f"test_tokens: {test_tokens}")

        if test_tokens <= max_tokens:
            trimmed_data.append(item)
            total_tokens = calculate_tokens(trimmed_data, encoder)
            #print(total_tokens)
        else:
            break

    # Restore original order
    trimmed_data.reverse()

    # If the first item's role is "user", keep the data; if "assistant", remove it
    if trimmed_data and trimmed_data[0]["role"] == "assistant":
        trimmed_data.pop(0)

    # Add system messages back to the beginning
    trimmed_data = system_messages + trimmed_data

    # Write results to a new JSON file
    #write_json(file_path, trimmed_data)

    #print(f"Processed file saved as: {output_file}")
    final_tokens = calculate_tokens(trimmed_data, encoder)
    if final_tokens != current_total_tokens:
        print(f"Total tokens after trimming: {final_tokens}")
        print(f"Items kept: {len(trimmed_data)} out of {len(data)}")
        print(f"Tokens removed: {current_total_tokens - final_tokens}")

    return trimmed_data


def occupy_llm(instance: LLMInterface):

    occupy_path = instance.occupy_path
    llm_choice = instance.llm_choice
    target_cmd = instance.project_id
    full_regions = instance.full_regions

    print(occupy_path)
    occupy_data = read_json(occupy_path)
    found = False
    # given_azure_endpoint = None
    # given_region = None

    for item in occupy_data[llm_choice]:
        if item['use'] is False:
            # if not ('par' in item and item['par'] == "3"):
            #     continue
            if 'region' in item and item['region'] in full_regions:
                continue
            given_api_key = item['given_api_key'] 
            if 'given_azure_endpoint' in item:
                given_azure_endpoint = item['given_azure_endpoint'] 
            if 'region' in item:
                given_region = item['region']
            if 'given_model' in item:
                given_model = item['given_model']
            if 'client_id' in item:
                given_client_id = item['client_id']

            item['use'] = True
            item['project_id'] = target_cmd
            found = True
            break
    if not found:
        raise ValueError("Did not find an empty llm instance.")
    
    write_json(occupy_path, occupy_data)

    instance.api_key = given_api_key
    instance.azure_endpoint = given_azure_endpoint
    instance.region = given_region
    instance.client_id = given_client_id

    if 'claude' in llm_choice:
        instance.llm_model = get_claude_model(llm_choice)
    # given_model = instance.model
    
    return instance



def configure_llm(instance: LLMInterface, given_api_key: str, given_azure_endpoint: str, given_model: str):

    occupy_path = instance.occupy_path
    llm_choice = instance.llm_choice
    target_cmd = instance.project_id
    full_regions = instance.full_regions

    instance.api_key = given_api_key
    instance.azure_endpoint = given_azure_endpoint
    instance.region = None
    instance.llm_model = given_model

    return instance



def shutdown_llm(instance: LLMInterface):
    
    if not isinstance(instance, LLMInterface):
        return False, f"Not an LLMInterface: {type(instance)}"

    given_api_key = instance.api_key
    given_azure_endpoint = instance.azure_endpoint
    given_region = instance.region
    occupy_path = instance.occupy_path
    llm_choice = instance.llm_choice
    token_path = instance.token_path
    
    occupy_data = read_json(occupy_path)
    found = False
    for item in occupy_data[llm_choice]:
        if item['given_api_key'] != given_api_key:
            continue
        if 'given_azure_endpoint' in item and item['given_azure_endpoint'] != given_azure_endpoint:
            continue

        item['use'] = False
        if 'region' in item:
            print(f"\nShutdown: {item['region']}, {given_api_key}, {given_azure_endpoint}")

            if os.path.exists(token_path):
                cost = calc_claude_cost_from_file(token_path)
                print(f"Cost: ${cost['total_cost_usd']:.2f}")
        #item['program'] = None
        found = True
        break

    write_json(occupy_path, occupy_data)


def fix_escapes(json_str):
    # Fix specific escape patterns
    patterns = {
        r'\n': r'\\n',
        r'\r': r'\\r',
        r'\"': r'\\"'
    }
    for old, new in patterns.items():
        json_str = json_str.replace(old, new)
    return json_str


def base64_decode(string):
    # Add padding if missing
    padding = 4 - (len(string) % 4) if len(string) % 4 else 0
    string = string + "=" * padding
    
    # Attempt to decode
    decoded = base64.b64decode(string)
    return decoded.decode('utf-8')


def is_base64_decodable(string):
    try:
        # Add padding if missing
        padding = 4 - (len(string) % 4) if len(string) % 4 else 0
        string = string + "=" * padding
        
        # Attempt to decode
        base64.b64decode(string).decode('utf-8')
        return True
    except Exception as e:
        return False


def extract_json_response(llm_choice, response_text):
    error_text = None
    decode_failure = False  # added

    if isinstance(response_text, dict):
        # If it is already a Python dictionary object
        response_json = response_text
    else:
        try:
            match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
            if match:
                response_text = match.group(1).strip()

            """ 
            # Detect and remove ```json ... ``` code block format
            if '```json' in response_text:
            # if response_text.strip().startswith('```'):
                # Remove the first ``` line
                lines = response_text.strip().split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                # Remove the last ``` line
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                response_text = '\n'.join(lines)
            """ 

            if llm_choice in ['gpt', 'gpt_azure', 'gpt_azure_databricks']:
                if response_text is None:
                    decode_failure = True
                    return decode_failure, error_text
                response_json = json.loads(response_text)

            elif llm_choice in ['claude', 'claude_azure', 'claude_bedrock']:
                # response_json = json.loads("{" + response_text[:response_text.rfind("}") + 1], strict=False)
                response_json = json.loads(response_text, strict=False)
                # response_json = json.loads("{" + response_text[:response_text.rfind("}") + 1])
                # if 'answer' in response_json and isinstance(response_json["answer"], list) and len(response_json["answer"]) > 0:
                #     if 'modified_data' in response_json["answer"][0]:
                #         try:
                #             modified_data_content = json.loads(response_json["answer"][0]["modified_data"])
                #         except json.JSONDecodeError as e:
                #                 print(f"Failed to decode modified_data JSON: {str(e)}")
                #                 decode_failure = True
                #                 return decode_failure

        except json.JSONDecodeError as e:
            print("Failed to decode JSON")
            print("------------------ response_text start ------------------")
            print(response_text)
            print("------------------ response_text end ------------------")
            print(f"Failed to decode modified_data JSON: {str(e)}")
            
            decode_failure = True
            error_text = f"Failed to decode modified_data JSON: {str(e)}"
            # raise ValueError("Failed to decode JSON.")
            try:
                response_text = fix_escapes(response_text)
                response_json = json.loads(response_text, strict=False)
                decode_failure = False

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                decode_failure = True
                error_text = f"Failed to decode modified_data JSON: {str(e)}"
                # raise ValueError("Failed to decode JSON.")


            """
            # rgba is particularly hard to decode
            decode_failure, cleaned_json = clean_response(response_text)
            if decode_failure is True:
                return decode_failure
            else:
                response_json = cleaned_json
                response_json['ongoing'] = True
            """

            return decode_failure, error_text

    code_blocks = []

    # Store JSON fields into a list
    if 'rust_code' in response_json:
        # decoded_rust_code = base64.b64decode(response_json['rust_code']).decode('utf-8')
        # response_json['rust_code'] = decoded_rust_code
        decoded_rust_code = response_json['rust_code']
            
        # Add padding (if necessary)
        padding = '=' * (-len(decoded_rust_code) % 4)
        decoded_rust_code = decoded_rust_code + padding
        
        # decoded_rust_code = base64.b64decode(decoded_rust_code).decode('utf-8')
        
        # Update JSON with decoded code
        # response_json['rust_code'] = decoded_rust_code
        if is_base64_decodable(response_json['rust_code']):
            response_json['rust_code'] = base64_decode(response_json['rust_code'])

        code_blocks.append(response_json['rust_code'])
        # print(f"rust_code: {response_json['rust_code']}")

    if 'toml' in response_json:
        code_blocks.append(response_json['toml'])
        # print(response_json['toml'])
    if 'build' in response_json:
        code_blocks.append(response_json['build'])
        # print(response_json['build'])
    if 'answer' in response_json:
        code_blocks.append(response_json['answer'])
        # print(f"answer: {response_json['answer']}")
    if 'code' in response_json:
        code_blocks.append(response_json['code'])
        # print(f"code: {response_json['code']}")
    if 'reason' in response_json:
        code_blocks.append(response_json['reason'])
        # print(response_json['reason'])
    if 'error_path' in response_json:
        code_blocks.append(response_json['error_path'])
    
    if 'max_counter' in response_json:
        global TESTFILE_COUNTER
        if response_json['max_counter'] is not None:
            TESTFILE_COUNTER = response_json['max_counter']
            TESTFILE_COUNTER += 1  # start of the next subset
            print(f"Counter updated: {TESTFILE_COUNTER}")

    if 'parsable' in response_json:
        code_blocks.append(response_json['parsable'])

        if not response_json['parsable']:
            print(response_json)
            raise ValueError("response_json is unparsable")

        print("----------------------------- error_path -----------------------------")
        print(response_json['error_path'])

    # else:
    #     print("Neither expected key (e.g., 'rust_code' nor 'toml') found in the response")
    print(f"------------------------ Response start ------------------------")
    print(f"Skipeed, because maybe too long.")
    # print(f"{response_json}")
    print(f"------------------------ Response end ---------------------------")  

    return response_json, error_text  # code_blocks



def calc_claude_cost_from_file(
    json_path: str,
    input_price_per_million: float = 5.0, #15.0,
    output_price_per_million: float = 25.0, #75.0,
) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    total_input_tokens = sum(int(r.get("input_token", 0) or 0) for r in records)
    total_output_tokens = sum(int(r.get("output_token", 0) or 0) for r in records)

    input_cost = total_input_tokens / 1_000_000 * input_price_per_million
    output_cost = total_output_tokens / 1_000_000 * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    }


def ask_llm(prompt: str, memory_type: str, llm_interface: LLMInterface = None) -> str:

    ################## Initial set ##################
    llm_choice = llm_interface.llm_choice
    output_max = llm_interface.output_max
    context_window = llm_interface.context_window
    temperature = llm_interface.temperature
    llm_model = llm_interface.llm_model
    client_id = llm_interface.client_id

    #llm_interface = config.llm_interface

    api_key = llm_interface.api_key

    given_model = llm_interface.llm_model
    given_azure_endpoint = llm_interface.azure_endpoint
    given_api_key = llm_interface.api_key
    given_temperature = llm_interface.temperature

    timeout = llm_interface.timeout

    #system_prompt = llm_interface.system_prompt
    history_path = llm_interface.history_path
    token_path = llm_interface.token_path
    database_dir = llm_interface.database_dir
    chat_dir = llm_interface.chat_dir
    count_path = llm_interface.count_path
    exp_data = llm_interface.exp_data

    # print("------------")
    # print(given_api_key)
    # print(given_azure_endpoint)
    # print(given_temperature)
    # print(llm_model)
    # print(llm_choice)
    # print(context_window)
    # print(output_max)
    # print("------------")

    ##################################################

    DEBUG_LLM = False

    prompt = clean_prompt(prompt)
    prompt = create_prompt_string(prompt) #"\n".join(prompt)
    print(f"------------------------ Prompt ------------------------")
    print(f"{prompt}")
    print(f"--------------------------------------------------------")
    write_prompt(database_dir, f"request", prompt, chat_dir, count_path)
    write_prompt(database_dir, f"user", prompt, chat_dir, count_path)

    code_blocks = []
    text = None
    input_token = None
    output_token = None
    chat_history = None

    if isinstance(memory_type, str) and memory_type == "init":
        chat_history = []
        # Add system message
        if llm_choice in ['claude_azure']: # ['claude', 
            chat_history.append({
                "role": "system",
                "content": "Always wrap your JSON response in ```json ... ``` markdown code blocks."
            })

    elif isinstance(memory_type, str) and memory_type == "continue":
        if not os.path.exists(history_path): # For starting with continue from the beginning
            write_json(history_path, [])
        # trim_json_data(llm_choice, history_path, 150000) # Hits the limit at 200000
        chat_history = read_json(history_path)
    else:
        raise ValueError("You need to set the memory_type: init or continue.")
    
    if chat_history is None:
        chat_history = []
        
    if len(chat_history) == 0:
        if llm_choice in ['claude_azure']: # ['claude', 
            chat_history.append({
                "role": "system",
                "content": "Always wrap your JSON response in ```json ... ``` markdown code blocks."
            })

    if llm_choice in ['gpt', 'gpt_azure', 'gpt_azure_databricks', 'claude', 'claude_azure', 'claude_bedrock']:
        chat_history.append({"role": "user", "content": prompt})
        
        if llm_choice in ['gpt', 'gpt_azure', 'gpt_azure_databricks']:
            trim_max = 100000
        elif llm_choice in ['claude', 'claude_azure', 'claude_bedrock']:
            trim_max = 180000 # 130000 # 150000

        # Need to trim here
        # tmp_history_path = f'{database_dir}/tmp_history.json'
        # write_json(tmp_history_path, chat_history)
        chat_history = trim_json_data(llm_choice, llm_model, chat_history, trim_max)
        # chat_history = read_json(tmp_history_path)
        # delete_file(tmp_history_path)
    
    elif llm_choice == 'Gemini':
        print("Getting gemini_history")
        #gemini_history_path = "gemini_history.json"
        #write_json(gemini_history_path, chat_history)
        gemini_history = transform_gemini_history(chat_history)
    
    elif llm_choice == 'Llama':
        print("Does not append for llama.")

    if os.path.exists(token_path):
        cost = calc_claude_cost_from_file(token_path)
        print(f"Current cost: ${cost['total_cost_usd']:.2f}")

    ########################################################
    if llm_choice == 'gpt':
        if 'repair_count' in exp_data and 'file_path' in exp_data:
            print(f"repair_count is {exp_data['repair_count']} for {exp_data['file_path']}")
            
        # Set OpenAI API key
        if not DEBUG_LLM:         
            openai.api_key = openai_api_key

            retries = 0
            max_retries = 10
            wait_time = 30
            response = None
            while retries < max_retries:
                try:
                    response = openai.ChatCompletion.create(
                        model=llm_model, #gpt_model,
                        messages=chat_history,
                        #messages=[
                        #    {"role": "system", "content": "You are a helpful assistant that returns JSON as a response."}, #"You are a helpful assistant."},
                        #    {"role": "user", "content": prompt}
                        #],
                        response_format={"type": "json_object"},
                        temperature=given_temperature, #0, #0.7,
                        max_tokens=output_max,  #4096,
                        #top_p=1.0,
                        #frequency_penalty=0.5,
                        #presence_penalty=0.5,
                    )
                    break  # Exit the loop on success
                    
                except openai.error.RateLimitError:
                    retries += 1
                    print(f"Rate limit hit. Waiting for {wait_time} seconds...")
                    time.sleep(wait_time)

            if response:  # Only process if the response was successful
                text = response.choices[0].message['content']
                print(text)
            else:
                print("Failed to get a response after multiple retries.")

            #code_blocks = extract_code_blocks(text)  # ask_llm function before JSON format
            while(1):
                code_blocks = extract_json_response(llm_choice, text)

                if isinstance(code_blocks, bool):
                    print("Failure in getting code_blocks.")

                    chat_history.append({"role": "system", "content": "Failure in getting correct JSON format."})
                    #chat_history.append({"role": "user", "content": "Make sure to properly escape control characters within the JSON string of the response. Also, to meet the token limit, please respond with 100 lines at a time."})
                    instruction = "Make sure to properly escape control characters within the JSON string of the response. Also, to meet the token limit, please respond with 100 lines at a time."
                    chat_history.append({"role": "user", "content": instruction})
                    write_prompt(database_dir, f"request", instruction, chat_dir, count_path)
                    # prompt_count = load_prompt_count()
                    write_prompt(database_dir, f"user", instruction, chat_dir, count_path)

                    #print(prompt)
                    response = None
                    while retries < max_retries:
                        try:
                            response = openai.ChatCompletion.create(
                                model=gpt_model,
                                messages=chat_history,
                                #messages=[
                                #    {"role": "system", "content": "You are a helpful assistant that returns JSON as a response."}, #"You are a helpful assistant."},
                                #    {"role": "user", "content": prompt}
                                #],
                                response_format={"type": "json_object"},
                                temperature=given_temperature, #0, #0.7,
                                max_tokens=output_max, #4096,
                                #top_p=1.0,
                                #frequency_penalty=0.5,
                                #presence_penalty=0.5,
                            )
                            break  # Exit the loop on success
                            
                        except openai.error.RateLimitError:
                            retries += 1
                            print(f"Rate limit hit. Waiting for {wait_time} seconds...")
                            time.sleep(wait_time)

                    if response:  # Only process if the response was successful
                        text = response.choices[0].message['content']
                        print(text)
                    else:
                        print("Failed to get a response after multiple retries.")
                    
                else:
                    print("Secceed in gettineg a correct format.")
                    break

            total_tokens = response['usage']['total_tokens']
            print(f"Total token in {total_tokens}")
            
            prompt_used = response['usage']['prompt_tokens']
            input_token = response['usage']['prompt_tokens']
            print(f"Prompt token in {prompt_used}")

            completion = response['usage']['completion_tokens']
            output_token = response['usage']['completion_tokens']
            print(f"Response token in {completion}")
              
        else:
            print("DEBUG_LLM Mode")


    elif llm_choice == 'gpt_azure':
        print("Asking gpt_azure...")

        given_api_version="2024-05-01-preview"  # Fixed
        given_api_version="2025-01-01-preview"

        if 'repair_count' in exp_data and 'file_path' in exp_data:
            print(f"repair_count is {exp_data['repair_count']} for {exp_data['file_path']}")
            
        # Set OpenAI API key
        if not DEBUG_LLM:         
            openai.api_key = openai_api_key

            retries = 0
            max_retries = 10
            wait_time = 30
            response = None
            while retries < max_retries:
                try:
                    client = AzureOpenAI(
                        api_version=given_api_version,
                        api_key=given_api_key,
                        azure_endpoint=given_azure_endpoint 
                    )
                    response = client.chat.completions.create( #response = openai.ChatCompletion.create(
                        model=llm_model, #gpt_model,
                        messages=chat_history,
                        #messages=[
                        #    {"role": "system", "content": "You are a helpful assistant that returns JSON as a response."}, #"You are a helpful assistant."},
                        #    {"role": "user", "content": prompt}
                        #],
                        response_format={"type": "json_object"},
                        temperature=given_temperature, #0, #0.7,
                        max_tokens=output_max, #4096,
                        #top_p=1.0,
                        #frequency_penalty=0.5,
                        #presence_penalty=0.5,
                    )
                    break  # Exit the loop on success
                    
                except openai.error.RateLimitError:
                    retries += 1
                    print(f"Rate limit hit. Waiting for {wait_time} seconds...")
                    time.sleep(wait_time)

            if response:  # Only process if the response was successful
                text = response.choices[0].message.content  #text = response.choices[0].message['content']
                print(text)

                total_tokens = response.usage.total_tokens #response['usage']['total_tokens']
                print(f"Total token in {total_tokens}")
                
                prompt_used = response.usage.prompt_tokens #response['usage']['prompt_tokens']
                input_token = response.usage.prompt_tokens #response['usage']['prompt_tokens']
                print(f"Prompt token in {prompt_used}")

                completion = response.usage.completion_tokens #response['usage']['completion_tokens']
                output_token = response.usage.completion_tokens #response['usage']['completion_tokens']
                print(f"Response token in {completion}")
                
                update_token(input_token, 0, token_path)

            else:
                print("Failed to get a response after multiple retries.")

            #code_blocks = extract_code_blocks(text)  # ask_llm function before JSON format
            while(1):
                code_blocks, error_text = extract_json_response(llm_choice, text)

                if isinstance(code_blocks, bool):                
                    print("Failure in getting code_blocks.")

                    #"""
                    # Write LLM response. Is this correct?
                    write_prompt(database_dir, f"response", code_blocks, chat_dir, count_path)
                    # prompt_count = load_prompt_count()
                    write_prompt(database_dir, f"llm", code_blocks, chat_dir, count_path)
                    #"""
                    update_token(0, output_token, token_path)
                    

                    chat_history.append({"role": "system", "content": "Failure in getting correct JSON format."})
                    #chat_history.append({"role": "user", "content": "Make sure to properly escape control characters within the JSON string of the response. Also, to meet the token limit, please respond with 100 lines at a time."})
                    instruction = "Make sure to properly escape control characters within the JSON string of the response. Also, to meet the token limit, please respond with 100 lines at a time."
                    chat_history.append({"role": "user", "content": instruction})
                    write_prompt(database_dir, f"request", instruction, chat_dir, count_path)
                    # prompt_count = load_prompt_count()
                    write_prompt(database_dir, f"user", instruction, chat_dir, count_path)

                    update_token(input_token, 0, token_path)

                    #print(prompt)
                    response = None
                    while retries < max_retries:
                        try:
                            client = AzureOpenAI(
                                api_version=given_api_version,
                                api_key=given_api_key,
                                azure_endpoint=given_azure_endpoint
                            )
                            response = client.chat.completions.create( #response = openai.ChatCompletion.create(
                                model=gpt_model,
                                messages=chat_history,
                                #messages=[
                                #    {"role": "system", "content": "You are a helpful assistant that returns JSON as a response."}, #"You are a helpful assistant."},
                                #    {"role": "user", "content": prompt}
                                #],
                                response_format={"type": "json_object"},
                                temperature=given_temperature, #0, #0.7,
                                max_tokens=output_max, #4096,
                                #top_p=1.0,
                                #frequency_penalty=0.5,
                                #presence_penalty=0.5,
                            )
                            break  # Exit the loop on success
                            
                        except openai.error.RateLimitError:
                            retries += 1
                            print(f"Rate limit hit. Waiting for {wait_time} seconds...")
                            time.sleep(wait_time)

                    if response:  # Only process if the response was successful
                        text = response.choices[0].message.content  #text = response.choices[0].message['content']
                        print(text)

                        total_tokens = response.usage.total_tokens #response['usage']['total_tokens']
                        print(f"Total token in {total_tokens}")
                        
                        prompt_used = response.usage.prompt_tokens #response['usage']['prompt_tokens']
                        input_token = response.usage.prompt_tokens #response['usage']['prompt_tokens']
                        print(f"Prompt token in {prompt_used}")

                        completion = response.usage.completion_tokens #response['usage']['completion_tokens']
                        output_token = response.usage.completion_tokens #response['usage']['completion_tokens']
                        print(f"Response token in {completion}")
                        
                        
                    else:
                        print("Failed to get a response after multiple retries.")
                    
                else:
                    print("Secceed in gettineg a correct format.")
                    break
        else:
            print("DEBUG_LLM Mode")

    elif llm_choice == 'gpt_azure_databricks':
        risky_error = 0
        # https://arunprakash.ai/posts/anthropic-claude3-messages-api-json-mode/messages_api_json.html
        # Commenting out for now as this causes errors
        if not DEBUG_LLM:
            # How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html

            long_count = 0
            delay=1
            while(1):
                print("Asking gpt_azure_databricks...")
                if 'repair_count' in exp_data and 'file_path' in exp_data:
                    print(f"repair_count is {exp_data['repair_count']} for {exp_data['file_path']}")

                client = OpenAI(
                    api_key=given_api_key,
                    base_url=given_azure_endpoint
                )
                print("============ chat_histroty start ============")
                #print(chat_history)
                print("Skipping chat history")
                print("============ chat_histroty end ============")
                max_retries = 5
                for attempt in range(max_retries):
                    response_flag = False
                    try: 
                        message = client.chat.completions.create(
                            model=llm_model, #given_model, #"databricks-claude-3-7-sonnet", #"claude-3-7-sonnet-20250219", #"claude-3-5-sonnet-20241022", #"claude-3-5-sonnet-20240620", #claude_model, # Model: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
                            max_tokens=output_max, #8192, #4096, # 4096 see below
                            #temperature=given_temperature, #0, # Higher values make it more chaotic # You are a helpful assistant that returns JSON as a response.
                            messages=chat_history,
                            #extra_headers = {"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                        )

                        response_flag = True

                    except openai.APIStatusError as e:
                        # https://ohina.work/post/azure_openai_error/
                        print(f"OpenAI APIStatus Error: [{e}]")

                        # 400: BadRequest
                        if type(e) is openai.BadRequestError:
                            print(f"OpenAI BadRequest Error: [{e}]")
                            # When the token count exceeds the context window maximum
                            if "This model's maximum context length is" in str(e):
                                print(f"Token over [{e}]")
                            # When blocked by the content filter
                            # Error code: 400 - {'error': {'message': "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766", 'type': None, 'param': 'prompt', 'code': 'content_filter', 'status': 400, 'innererror': {'code': 'ResponsibleAIPolicyViolation', 'content_filter_result': {'hate': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': True, 'severity': 'medium'}}}}}
                            elif "The response was filtered due to the prompt triggering Azure OpenAI's content management policy." in str(e):
                                content_filter_result = str(e).split("content_filter_result': ")[1].split("}}")[0].replace("'", '"') + "}}"
                                content_filter_result = content_filter_result.replace("True", "true").replace("False", "false")
                                print(f"Content filter result: [{content_filter_result}]")
                                json_content_filter_result = json.loads(content_filter_result)
                                for key, value in json_content_filter_result.items():
                                    if value['filtered']:
                                        print(f"Content filter result: [{key}] : [{value}]")

                            else:
                                print(f"Content filter error ?")

                        # 401 Unauthorized. Access token is missing, invalid, audience is incorrect
                        if type(e) is openai.AuthenticationError:
                            print(f"OpenAI Authentication Error: [{e}]")
                        # 403: Permission Denied
                        elif type(e) is openai.PermissionDeniedError:
                            print(f"OpenAI Permission Denied Error: [{e}]")
                        # 404: Not Found
                        elif type(e) is openai.NotFoundError:
                            print(f"OpenAI NotFound Error: [{e}]")
                        # 408: Operation Timeout
                        # openai.APIStatusError: Error code: 408 - {'error': {'code': 'Timeout', 'message': 'The operation was timeout.'}}
                        elif "The operation was timeout." in str(e):
                            print(f"OpenAI Timeout Error: [{e}]")
                        # 409: Conflict
                        elif type(e) is openai.ConflictError:
                            print(f"OpenAI Conflict Error: [{e}]")
                        # 422: Unprocessable Entity
                        elif type(e) is openai.UnprocessableEntityError:
                            print(f"OpenAI Unprocessable Entity Error: [{e}]")
                        # 429: Rate Limit
                        elif type(e) is openai.RateLimitError:
                            print(f"OpenAI Rate Limit Error: [{e}]")
                            # str(e) -> Error code: 429 - {'error': {'code': '429', 'message': 'Requests to the ChatCompletions_Create Operation under Azure OpenAI API version 2024-05-01-preview have exceeded token rate limit of your current OpenAI S0 pricing tier. Please retry after 58 seconds. Please go here: https://aka.ms/oai/quotaincrease if you would like to further increase the default rate limit.'}}
                            # get wait time from error message
                            wait_time = int(str(e).split("Please retry after ")[1].split(" seconds.")[0])
                            print(f"Rate Limit Error: Wait time: {wait_time}")
                        # 500: Internal Server Error
                        elif type(e) is openai.InternalServerError:
                            print(f"OpenAI Internal Server Error: [{e}]")



                    except InternalServerError as e:
                        if attempt == max_retries - 1:
                            raise  # Re-raise the exception if max retries reached
                        print(f"InternalServerError occurred. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff: double the wait time

                    if response_flag:
                        break
            
                text = message.choices[0].message.content #message.content[0].text
                print(text)

                # input_token = message.usage.input_tokens
                # output_token = message.usage.output_tokens
                # print(message.usage.input_tokens)
                # print(message.usage.output_tokens)

                total_tokens = message.usage.total_tokens #response['usage']['total_tokens']
                print(f"Total token in {total_tokens}")
                
                input_token = message.usage.prompt_tokens #response['usage']['prompt_tokens']
                print(f"Prompt token in {input_token}")

                output_token = message.usage.completion_tokens #response['usage']['completion_tokens']
                print(f"Response token in {output_token}")
        

                update_token(input_token, 0, token_path)

                code_blocks, error_text = extract_json_response(llm_choice, text)
                if code_blocks is not True:
                    break
                else:
                    # prompt_count = load_state()
                    write_prompt(database_dir, f"llm", code_blocks, chat_dir, count_path)
                    
                    update_token(0, output_token, token_path)
                    #chat_history.append({"role": "assistant", "content": "Too long response"})
                    #chat_history.append({"role": "user", "content": f"The answer length exceeds 4096 tokens. Please make sure it is {output_max} tokens or less and respond again. Return only JSON format data without including any text."})

                    if output_token > output_max: #4000: # Already hit the limit at 4073
                        print("Too long response")
                        chat_history.append({"role": "assistant", "content": "Too long response"})

                        print(f"Analyzing {long_count}")
                        # 20 lines (or more) didn't work here. Interesting! Haven't tested the range between 10-20 lines though.
                        addition = f"""The answer exceeds {output_max} tokens in length.
When including code in the response, even if it's in the middle of a logical unit (function, data structure, etc.), please divide the code in the JSON key into chunks of 100 lines segments and answer the first segment now. Please make sure not to truncate the JSON data in your response.
Also, if there is remaining code, set the value of the 'ongoing' key to a boolean value of True. If the code is the final part, set the value of the 'ongoing' key to a boolean value of False.
"""

                        write_prompt(database_dir, f"request", addition, chat_dir, count_path)
                        # prompt_count = load_prompt_count()
                        write_prompt(database_dir, f"user", addition, chat_dir, count_path)

                        chat_history.append({"role": "user", "content": f"{addition}"})
                        
                        long_count += 1
                    else:
                        print("Imappropriate response format")
                        chat_history.append({"role": "assistant", "content": "Imappropriate response format"})
                        #chat_history.append({"role": "user", "content": f"The response in JSON format could not be correctly JSON decoded. If Rust code is included in the response, please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. Return only JSON format data without including any text."})

                        if error_text is None:
                            print("Error in addition_error1")
                            addition_error1 = f"The response in JSON format could not be decoded correctly. Please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure.  When representing backslashes as byte literals, escape the backslash twice in the source code, and also escape it again in the byte literal, resulting in four backslashes (double backslashes). When representing backslashes as character literals, escape the backslash once in the source code and again in the character literal, resulting in two backslashes. Respond again with only JSON data and no text."
                            chat_history.append({"role": "user", "content": f"{addition_error1}"})
                            # prompt_count = load_state()
                            write_prompt(database_dir, f"user", addition_error1, chat_dir, count_path)

                        else:
                            if risky_error > 10:
                                raise ValueError("Stop due to bad format.")
                            print("Error in addition_error2")
                            #addition_error2 = f"The response in JSON format could not be decoded correctly. Please respond again with only one JSON data and no text so that it can be processed with json.loads() and also please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. Please include the explanation text in the reason text within the JSON data. Please respond with only one pure JSON data without using code blocks or markdown syntax such as ```json <text>```."
                            addition_error2 = f"The response in JSON format could not be decoded correctly. Please wrap your JSON response in ```json ... ``` markdown code blocks. Properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. Please include the explanation text in the reason field within the JSON data."
                            #please encode the rust_code included in your response using base64.b64encode(text.encode('utf-8')), and include it as the value of 'rust_code' in JSON format."
                            chat_history.append({"role": "user", "content": f"{addition_error2}"})
                        
                            # prompt_count = load_state()
                            write_prompt(database_dir, f"user", addition_error2, chat_dir, count_path)

                            risky_error += 1


    elif llm_choice == 'Gemini':
        if 'repair_count' in exp_data:
            print(f"repair_count is {exp_data['repair_count']} for {exp_data['file_path']}")
            
        # configure
        genai.configure(api_key=gemini_api_key)
        gemini_chat_model = genai.GenerativeModel(model_name=gemini_model) #"gemini-pro")

        # add history
        gemini_chat = gemini_chat_model.start_chat(history=gemini_history)

        # send message
        max_attempts = 10  # Maximum number of attempts
        max_decode_attempts = 10  # Maximum number of decode attempts
        attempt = 0
        decode_attempt = 0
        wait_time = 10
        while attempt < max_attempts:
            try:
                while(1):

                    if decode_attempt >= max_decode_attempts:
                        break
    
                    response = gemini_chat.send_message(prompt)
                    code_blocks = extract_braces_json(response.text)

                    if code_blocks is None:
                        print("Failure in getting code_blocks.")
                        #prompt = prompt + "Make sure to properly escape control characters within the JSON string of the response."
                        instruction = "Make sure to properly escape control characters within the JSON string of the response."
                        prompt = prompt + instruction
                        print(prompt)
                        write_prompt(database_dir, f"request", instruction, chat_dir, count_path)
                        # prompt_count = load_prompt_count()
                        write_prompt(database_dir, f"user", instruction, chat_dir, count_path)

                        response = gemini_chat.send_message(prompt)
                        code_blocks = extract_braces_json(response.text)

                        print(f"decode_attempt sttempting {decode_attempt}")
                        decode_attempt += 1

                    else:
                        print("Succeed in getting code_blocks")
                        input_token = response.usage_metadata.prompt_token_count #gemini_chat_model.generate_content(prompt)
                        print(f"input_token: {input_token}")

                        output_token = response.usage_metadata.candidates_token_count #gemini_chat_model.generate_content(response)
                        print(f"output_token: {output_token}")
                        break
                break

            except Exception as e:
                print(f"An error occurred: {e}")
                time.sleep(wait_time)
                attempt += 1
    
    
    elif llm_choice == 'Llama':

        # Fix this as the input causes issues
        os.environ["REPLICATE_API_TOKEN"] = llama_api_key

        while(1):
            error = False

            # Need to trim here
            tmp_history_path = f'{database_dir}/tmp_history.json'
            write_json(tmp_history_path, chat_history)
            trim_json_data(tmp_history_path, 4000)
            chat_history = read_json(tmp_history_path)
            delete_file(tmp_history_path)
            
            #formatted_history = "\n".join([f"user: {turn['user']}\nassistant: {turn['assistant']}" for turn in chat_history])
            formatted_history = "\n".join([f"user: {turn['content'] if turn['role'] == 'user' else ''}\nassistant: {turn['content'] if turn['role'] == 'assistant' else ''}" for turn in chat_history])
            prompt = f"{formatted_history}\nuser: {prompt}" # \nassistant:


            # Prompt length (14081) exceeds maximum input length (8096)"}
            input = {
                "top_k": 50,
                "top_p": 0.9,
                "prompt": prompt,
                "temperature": given_temperature,  # 0.75,
                "max_new_tokens": 8096, #4096, # RuntimeError: {"detail":"E1002 PromptTooLong: Prompt length (14592) exceeds maximum input length (8096)"}
                "min_new_tokens": -1
            }

            response_output = ""
            
            try:
                client = replicate.Client(api_token=llama_api_key, timeout=60)
                for event in replicate.stream(llama_model, input=input):
                    output = f"{event}"
                    response_output += output
                
                print("--------------------- response_output for llama ---------------------")
                print(response_output)
                print("--------------------- response_output end for llama ---------------------")

                code_blocks = extract_braces_json(response_output)

                if code_blocks is None:
                    error = True
                    print("Failure in getting code_blocks.")
                    chat_history.append({"role": "assistant", "content": "Imappropriate response format"})
                    #chat_history.append({"role": "user", "content": "The response in JSON format could not be correctly JSON decoded. If Rust code is included in the response, please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. Return only JSON format data without including any text."})
                    chat_history.append({"role": "user", "content": "The response in JSON format cannot be correctly JSON decoded. Please do not surround the JSON content with backticks. Please return the response in JSON format only, without including any text."})  # If Rust code is included in the response, escape characters (e.g., newlines, double quotes) must be properly escaped while maintaining the original JSON structure. 

            except RuntimeError as e:
                print(f"An error occurred with the Llama model: {str(e)}")
                error = True
                # Log the error
                addition = f"""The answer exceeds {output_max} tokens in length.
When including code in the response, even if it's in the middle of a logical unit (function, data structure, etc.), please divide the code in the JSON key into chunks of 100 lines answer the first segment now. Please make sure not to truncate the JSON data in your response.
Also, if there is remaining code, set the value of the 'ongoing' key to a boolean value of True. If the code is the final part, set the value of the 'ongoing' key to a boolean value of False.
"""
                chat_history.append({"role": "user", "content": f"{addition}"})
                
            if error is not True:
                break

        # Limiting to user and assistant roles here.
        #chat_history.append({"user": prompt})
        #chat_history.append({"assistant" : response_output})
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": response_output})
        
    elif llm_choice == 'claude':
        risky_error = 0
        # https://arunprakash.ai/posts/anthropic-claude3-messages-api-json-mode/messages_api_json.html
        # Commenting out for now as this causes errors
        if not DEBUG_LLM:

            long_count = 0
            delay=1
            while(1):
                print("Asking Claude Anthropic...")
                if 'repair_count' in exp_data and 'file_path' in exp_data:
                    print(f"repair_count is {exp_data['repair_count']} for {exp_data['file_path']}")
                client = anthropic.Anthropic(
                    api_key = given_api_key,
                )
                print("============ chat_histroty start ============")
                #print(chat_history)
                print("Skipping chat history")
                print("============ chat_histroty end ============")
                max_retries = 5
                for attempt in range(max_retries):
                    response_flag = False
                    try: 
                        with client.messages.stream(
                            model=llm_model,
                            max_tokens=output_max, #32000,
                            temperature=given_temperature,
                            #system="You are an assistant that responds only in JSON format. Adhere strictly to the JSON format, and when inserting code into the specified key values, include the code as a string. Also, properly escape characters that require escaping (e.g., newlines, double quotes).",
                            system="You are an assistant that responds in JSON format. Wrap your JSON response in ```json ... ``` markdown code blocks. Adhere strictly to the JSON format, and when inserting code into the specified key values, include the code as a string. Also, properly escape characters that require escaping (e.g., newlines, double quotes).",
                            messages=chat_history,
                        ) as stream:
                            message = stream.get_final_message()
                        
                        """
                        message = client.messages.create(
                            model=llm_model, #"claude-sonnet-4-20250514", #  "claude-3-7-sonnet-20250219", #"claude-3-5-sonnet-20241022", #"claude-3-5-sonnet-20240620", #claude_model, # Model: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
                            max_tokens=32000, #8192, #4096, # 4096 see below
                            temperature=given_temperature, #0, # Higher values make it more chaotic # You are a helpful assistant that returns JSON as a response.
                            #system="You are an assistant that responds only in JSON format. Responses must strictly follow the JSON format, and when inserting code into the specified key values, include the code as a string. Also, properly escape characters that require escaping (e.g., newlines, double quotes).", #system="You are an assistant that responds only in JSON format. Responses should always be returned in JSON format with values in the specified keys. If the value for a key contains code, escape it appropriately.", # Prompt (optional) # Responses should always be in the format {\"specified_key\": \"enter response here\"}.
                            system= "You are an assistant that responds only in JSON format. Adhere strictly to the JSON format, and when inserting code into the specified key values, include the code as a string. Also, properly escape characters that require escaping (e.g., newlines, double quotes).",
                            messages=chat_history,
                            extra_headers = {"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                            #messages=[
                            #    {"role": "user", "content": prompt},
                            #    #{"role":"assistant", "content": "Here is the JSON requested:\n{"}
                            #]
                        )
                        """

                        response_flag = True
                    except InternalServerError as e:
                        if attempt == max_retries - 1:
                            raise  # Re-raise the exception if max retries reached
                        print(f"InternalServerError occurred. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff: double the wait time

                    if response_flag:
                        break
            
                text = message.content[0].text
                print(text)

                input_token = message.usage.input_tokens
                output_token = message.usage.output_tokens
                print(message.usage.input_tokens)
                print(message.usage.output_tokens)

                update_token(input_token, 0, token_path)

                code_blocks, error_text = extract_json_response(llm_choice, text)
                if code_blocks is not True:
                    break
                else:
                    # prompt_count = load_state()
                    write_prompt(database_dir, f"llm", code_blocks, chat_dir, count_path)
                    
                    update_token(0, output_token, token_path)
                    #chat_history.append({"role": "assistant", "content": "Too long response"})
                    #chat_history.append({"role": "user", "content": f"The answer length exceeds 4096 tokens. Please make sure it is {output_max} tokens or less and respond again. Return only JSON format data without including any text."})

                    if output_token > output_max: #30000: #4000: # Already hit the limit at 4073
                        print("Too long response")
                        chat_history.append({"role": "assistant", "content": "Too long response"})

                        print(f"Analyzing {long_count}")
                        # 20 lines (or more) didn't work here. Interesting! Haven't tested the range between 10-20 lines though.
                        addition = f"""The answer exceeds {output_max} tokens in length.
When including code in the response, even if it's in the middle of a logical unit (function, data structure, etc.), please divide the code in the JSON key into chunks of 100 lines segments and answer the first segment now. Please make sure not to truncate the JSON data in your response.
Also, if there is remaining code, set the value of the 'ongoing' key to a boolean value of True. If the code is the final part, set the value of the 'ongoing' key to a boolean value of False.
"""
                        # When splitting, set the value of the 'parsable' key in the JSON response to false.
                        #print(prompt)
                        # Responses must always be strictly in JSON format only, without any explanatory text or additional text.
                        # If a single unit that can be parsed by ctags, such as a function or data type starting from a line with no indentation, is too long, split it midway.
                        # Make sure it is {output_max} tokens or less and respond again.
                        # When splitting midway, enter the boolean value False for the 'parsable' key value.
                        # Return only JSON format data without including any text.

                        write_prompt(database_dir, f"request", addition, chat_dir, count_path)
                        # prompt_count = load_prompt_count()
                        write_prompt(database_dir, f"user", addition, chat_dir, count_path)

                        chat_history.append({"role": "user", "content": f"{addition}"})
                        #chat_history.append({"role": "user", "content": f"The answer length exceeds 4096 tokens. Please make sure it is {output_max} tokens or less and respond again. Return only JSON format data without including any text."})
                        
                        long_count += 1
                    else:
                        print("Imappropriate response format")
                        chat_history.append({"role": "assistant", "content": "Imappropriate response format"})
                        #chat_history.append({"role": "user", "content": f"The response in JSON format could not be correctly JSON decoded. If Rust code is included in the response, please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. Return only JSON format data without including any text."})
                        #chat_history.append({"role": "user", "content": f"The response in JSON format could not be decoded correctly. Please do not surround the JSON content with backticks. Respond again with only JSON data and no text."})

                        if error_text is None:
                            print("Error in addition_error1")
                            #addition_error1 = f"The response in JSON format could not be decoded correctly. Please respond again with only one JSON data and no text so that it can be processed with json.loads() and also please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. Please include the explanation text in the reason text within the JSON data. Please respond with only one pure JSON data without using code blocks or markdown syntax such as ```json <text>```."
                            addition_error1 = f"The response in JSON format could not be decoded correctly. Please wrap your JSON response in ```json ... ``` markdown code blocks. Properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. When representing backslashes as byte literals, use four backslashes. When representing backslashes as character literals, use two backslashes."
                            #addition_error1 = f"The response in JSON format could not be decoded correctly. If Rust code is included in the response, please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure.  When representing backslashes as byte literals, escape the backslash twice in the source code, and also escape it again in the byte literal, resulting in four backslashes (double backslashes). When representing backslashes as character literals, escape the backslash once in the source code and again in the character literal, resulting in two backslashes. Respond again with only JSON data and no text."
                            chat_history.append({"role": "user", "content": f"{addition_error1}"})
                            # prompt_count = load_state()
                            write_prompt(database_dir, f"user", addition_error1, chat_dir, count_path)

                        else:
                            if risky_error > 10:
                                raise ValueError("Stop due to bad format.")
                            print("Error in addition_error2")
                            #chat_history.append({"role": "user", "content": f"The response in JSON format could not be decoded correctly. Please base64 encode the rust_code in the response using encoded = base64.b64encode(text.encode('utf-8')), and return it as the value of \"rust_code\" in JSON format."})
                            #addition_error2 = f"The response in JSON format could not be decoded correctly. Please respond again with only one JSON data and no text so that it can be processed with json.loads() and also please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. Please include the explanation text in the reason text within the JSON data. Please respond with only one pure JSON data without using code blocks or markdown syntax such as ```json <text>```."
                            addition_error2 = f"The response in JSON format could not be decoded correctly. Please wrap your JSON response in ```json ... ``` markdown code blocks. Properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. Please include the explanation text in the reason field within the JSON data."
                            #addition_error2 = f"The response in JSON format could not be decoded correctly. If Rust code is not included in the response, then please respond again with only JSON data and no text and also please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. If Rust code ('rust_code' key value) is included, for the rust_code field value, please use base64-encoded Rust code (using base64.b64encode(text.encode('utf-8'))). Do not use the raw Rust code containing escape characters and line breaks."
                            #please encode the rust_code included in your response using base64.b64encode(text.encode('utf-8')), and include it as the value of 'rust_code' in JSON format."
                            chat_history.append({"role": "user", "content": f"{addition_error2}"})
                        
                            # prompt_count = load_state()
                            write_prompt(database_dir, f"user", addition_error2, chat_dir, count_path)

                            risky_error += 1
    
    elif llm_choice == 'claude_azure':
        risky_error = 0
        # https://arunprakash.ai/posts/anthropic-claude3-messages-api-json-mode/messages_api_json.html
        # Commenting out for now as this causes errors
        if not DEBUG_LLM:

            # How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
            long_count = 0
            delay=1
            while(1):
                print("Asking Azure Claude Anthropic...")
                if exp_data is not None and 'repair_count' in exp_data and 'file_path' in exp_data:
                    print(f"repair_count is {exp_data['repair_count']} for {exp_data['file_path']}")

                ####
                """
                host = given_azure_endpoint
                client_id = client_id
                client_secret = given_api_key

                host_url = host.replace("/serving-endpoints", "")
                wc = WorkspaceClient(
                    host=host_url,
                    client_id=client_id,
                    client_secret=client_secret
                )

                token = wc.config.authenticate()["Authorization"].replace("Bearer ", "")

                client = OpenAI(
                    api_key=token,
                    base_url=f"{host}"
                )
                """
                ####

                client = OpenAI(
                    api_key=given_api_key,
                    base_url=given_azure_endpoint 
                )
                

                print("============ chat_histroty start ============")
                #print(chat_history)
                print("Skipping chat history")
                print("============ chat_histroty end ============")
                max_retries = 100 #5
            
                for attempt in range(max_retries):
                    response_flag = False
                    try: 
                        message = client.chat.completions.create(
                            model=llm_model, #"databricks-claude-sonnet-4", #"databricks-claude-3-7-sonnet", #"claude-3-7-sonnet-20250219", #"claude-3-5-sonnet-20241022", #"claude-3-5-sonnet-20240620", #claude_model, # Model: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
                            max_tokens=output_max, #8192, #4096, 
                            temperature=given_temperature, #0, # Higher values make it more chaotic # You are a helpful assistant that returns JSON as a response.
                            #system= "You are an assistant that responds only in JSON format. Adhere strictly to the JSON format, and when inserting code into the specified key values, include the code as a string. Also, properly escape characters that require escaping (e.g., newlines, double quotes).",
                            messages=chat_history,
                            #extra_headers = {"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                            #messages=[
                            #    {"role": "user", "content": prompt},
                            #    #{"role":"assistant", "content": "Here is the JSON requested:\n{"}
                            #]
                        )

                        response_flag = True

                    except openai.APIStatusError as e:
                        # https://ohina.work/post/azure_openai_error/
                        print(f"OpenAI APIStatus Error: [{e}]")

                        # 400: BadRequest
                        if type(e) is openai.BadRequestError:
                            print(f"OpenAI BadRequest Error: [{e}]")
                            # When the token count exceeds the context window maximum
                            if "This model's maximum context length is" in str(e):
                                print(f"Token over [{e}]")
                            # When blocked by the content filter
                            # Error code: 400 - {'error': {'message': "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766", 'type': None, 'param': 'prompt', 'code': 'content_filter', 'status': 400, 'innererror': {'code': 'ResponsibleAIPolicyViolation', 'content_filter_result': {'hate': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': True, 'severity': 'medium'}}}}}
                            elif "The response was filtered due to the prompt triggering Azure OpenAI's content management policy." in str(e):
                                content_filter_result = str(e).split("content_filter_result': ")[1].split("}}")[0].replace("'", '"') + "}}"
                                content_filter_result = content_filter_result.replace("True", "true").replace("False", "false")
                                print(f"Content filter result: [{content_filter_result}]")
                                json_content_filter_result = json.loads(content_filter_result)
                                for key, value in json_content_filter_result.items():
                                    if value['filtered']:
                                        print(f"Content filter result: [{key}] : [{value}]")

                            else:
                                print(f"Content filter error ?")

                        # 401 Unauthorized. Access token is missing, invalid, audience is incorrect
                        if type(e) is openai.AuthenticationError:
                            print(f"OpenAI Authentication Error: [{e}]")
                        # 403: Permission Denied
                        elif type(e) is openai.PermissionDeniedError:
                            print(f"OpenAI Permission Denied Error: [{e}]")
                        # 404: Not Found
                        elif type(e) is openai.NotFoundError:
                            print(f"OpenAI NotFound Error: [{e}]")
                        # 408: Operation Timeout
                        # openai.APIStatusError: Error code: 408 - {'error': {'code': 'Timeout', 'message': 'The operation was timeout.'}}
                        elif "The operation was timeout." in str(e):
                            print(f"OpenAI Timeout Error: [{e}]")
                        # 409: Conflict
                        elif type(e) is openai.ConflictError:
                            print(f"OpenAI Conflict Error: [{e}]")
                        # 422: Unprocessable Entity
                        elif type(e) is openai.UnprocessableEntityError:
                            print(f"OpenAI Unprocessable Entity Error: [{e}]")
                        # 429: Rate Limit
                        elif type(e) is openai.RateLimitError:
                            print(f"OpenAI Rate Limit Error: [{e}]")
                            # str(e) -> Error code: 429 - {'error': {'code': '429', 'message': 'Requests to the ChatCompletions_Create Operation under Azure OpenAI API version 2024-05-01-preview have exceeded token rate limit of your current OpenAI S0 pricing tier. Please retry after 58 seconds. Please go here: https://aka.ms/oai/quotaincrease if you would like to further increase the default rate limit.'}}
                            # get wait time from error message
                            # wait_time = int(str(e).split("Please retry after ")[1].split(" seconds.")[0])
                            # print(f"Rate Limit Error: Wait time: {wait_time}")

                            if "REQUEST_LIMIT_EXCEEDED" in str(e):
                                wait_time = 80 #65  # Wait 60 seconds since it's a per-minute limit
                                print(f"** Databricks Rate Limit: Waiting {wait_time} seconds...")
                                time.sleep(wait_time)
                                continue  # Continue the retry loop
                                
                        # 500: Internal Server Error
                        elif type(e) is openai.InternalServerError:
                            print(f"OpenAI Internal Server Error: [{e}]")



                    except InternalServerError as e:
                        if attempt == max_retries - 1:
                            raise  # Re-raise the exception if max retries reached
                        print(f"InternalServerError occurred. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff: double the wait time

                    if response_flag:
                        break
                
                if not response_flag:
                    raise ValueError(f"API request failed after {max_retries} retries")
            
                text = message.choices[0].message.content #message.content[0].text
                print(text)

                # input_token = message.usage.input_tokens
                # output_token = message.usage.output_tokens
                # print(message.usage.input_tokens)
                # print(message.usage.output_tokens)

                total_tokens = message.usage.total_tokens #response['usage']['total_tokens']
                print(f"Total token in {total_tokens}")
                
                input_token = message.usage.prompt_tokens #response['usage']['prompt_tokens']
                print(f"Prompt token in {input_token}")

                output_token = message.usage.completion_tokens #response['usage']['completion_tokens']
                print(f"Response token in {output_token}")
        

                update_token(input_token, 0, token_path)

                code_blocks, error_text = extract_json_response(llm_choice, text)
                if code_blocks is not True:
                    break
                else:
                    # prompt_count = load_state()
                    write_prompt(database_dir, f"llm", code_blocks, chat_dir, count_path)
                    
                    update_token(0, output_token, token_path)
                    #chat_history.append({"role": "assistant", "content": "Too long response"})
                    #chat_history.append({"role": "user", "content": f"The answer length exceeds 4096 tokens. Please make sure it is {output_max} tokens or less and respond again. Return only JSON format data without including any text."})

                    if output_token > output_max: #4000: # Already hit the limit at 4073
                        print("Too long response")
                        chat_history.append({"role": "assistant", "content": "Too long response"})

                        # print(f"Analyzing {long_count}")
                        # 20 lines (or more) didn't work here. Interesting! Haven't tested the range between 10-20 lines though.
                        addition = f"""The answer exceeds {output_max} tokens in length.
When including code in the response, even if it's in the middle of a logical unit (function, data structure, etc.), please divide the code in the JSON key into chunks of 100 lines segments and answer the first segment now. Please make sure not to truncate the JSON data in your response.
Also, if there is remaining code, set the value of the 'ongoing' key to a boolean value of True. If the code is the final part, set the value of the 'ongoing' key to a boolean value of False.
"""
                        # When splitting, set the value of the 'parsable' key in the JSON response to false.
                        #print(prompt)
                        # Responses must always be strictly in JSON format only, without any explanatory text or additional text.
                        # If a single unit that can be parsed by ctags, such as a function or data type starting from a line with no indentation, is too long, split it midway.
                        # Make sure it is {output_max} tokens or less and respond again.
                        # When splitting midway, enter the boolean value False for the 'parsable' key value.
                        # Return only JSON format data without including any text.

                        write_prompt(database_dir, f"request", addition, chat_dir, count_path)
                        # prompt_count = load_prompt_count()
                        write_prompt(database_dir, f"user", addition, chat_dir, count_path)

                        chat_history.append({"role": "user", "content": f"{addition}"})
                        #chat_history.append({"role": "user", "content": f"The answer length exceeds 4096 tokens. Please make sure it is {output_max} tokens or less and respond again. Return only JSON format data without including any text."})
                        
                        long_count += 1
                    else:
                        print("Imappropriate response format")
                        chat_history.append({"role": "assistant", "content": "Imappropriate response format"})
                        #chat_history.append({"role": "user", "content": f"The response in JSON format could not be correctly JSON decoded. If Rust code is included in the response, please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. Return only JSON format data without including any text."})
                        #chat_history.append({"role": "user", "content": f"The response in JSON format could not be decoded correctly. Please do not surround the JSON content with backticks. Respond again with only JSON data and no text."})

                        if error_text is None:
                            print("Error in addition_error1")
                            addition_error1 = f"The response in JSON format could not be decoded correctly. Please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure.  When representing backslashes as byte literals, escape the backslash twice in the source code, and also escape it again in the byte literal, resulting in four backslashes (double backslashes). When representing backslashes as character literals, escape the backslash once in the source code and again in the character literal, resulting in two backslashes. Respond again with only JSON data and no text."
                            chat_history.append({"role": "user", "content": f"{addition_error1}"})
                            # prompt_count = load_state()
                            write_prompt(database_dir, f"user", addition_error1, chat_dir, count_path)

                        else:
                            if risky_error > 10:
                                raise ValueError("Stop due to bad format.")
                            print("Error in addition_error2")
                            #chat_history.append({"role": "user", "content": f"The response in JSON format could not be decoded correctly. Please base64 encode the rust_code in the response using encoded = base64.b64encode(text.encode('utf-8')), and return it as the value of \"rust_code\" in JSON format."})
                            #addition_error2 = f"The response in JSON format could not be decoded correctly. Please respond again with only one JSON data and no text so that it can be processed with json.loads() and also please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. Please include the explanation text in the reason text within the JSON data. Please respond with only one pure JSON data without using code blocks or markdown syntax such as ```json <text>```."
                            addition_error2 = f"The response in JSON format could not be decoded correctly. Please wrap your JSON response in ```json ... ``` markdown code blocks. Properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. Please include the explanation text in the reason field within the JSON data."
                            #please encode the rust_code included in your response using base64.b64encode(text.encode('utf-8')), and include it as the value of 'rust_code' in JSON format."
                            chat_history.append({"role": "user", "content": f"{addition_error2}"})
                        
                            # prompt_count = load_state()
                            write_prompt(database_dir, f"user", addition_error2, chat_dir, count_path)

                            risky_error += 1
    
    elif llm_choice == 'claude_bedrock':
        risky_error = 0
        # https://arunprakash.ai/posts/anthropic-claude3-messages-api-json-mode/messages_api_json.html
        # Commenting out for now as this causes errors
        if not DEBUG_LLM:
            # How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
            long_count = 0
            delay=1
            while(1):
                print("Asking Bedrock Claude Anthropic...")
                if 'repair_count' in exp_data and 'file_path' in exp_data:
                    print(f"repair_count is {exp_data['repair_count']} for {exp_data['file_path']}")

                client = AnthropicBedrock(
                    # api_key=given_api_key,
                    # base_url=given_azure_endpoint
                    aws_access_key=given_api_key,
                    aws_secret_key=given_azure_endpoint,
                    aws_region=given_region, #"us-west-2",
                )
                print("============ chat_histroty start ============")
                #print(chat_history)
                print("Skipping chat history")
                print("============ chat_histroty end ============")
                max_retries = 5
                for attempt in range(max_retries):
                    response_flag = False
                    try: 
                        message = client.messages.create( #client.chat.completions.create(
                            model=llm_model, #"us.anthropic.claude-3-7-sonnet-20250219-v1:0", #"claude-3-7-sonnet-20250219", #"claude-3-5-sonnet-20241022", #"claude-3-5-sonnet-20240620", #claude_model, # Model: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
                            max_tokens=output_max, #8192, #4096, # 4096 see below
                            temperature=given_temperature, #0, # Higher values make it more chaotic # You are a helpful assistant that returns JSON as a response.
                            #system="You are an assistant that responds only in JSON format. Responses must strictly follow the JSON format, and when inserting code into the specified key values, include the code as a string. Also, properly escape characters that require escaping (e.g., newlines, double quotes).", #system="You are an assistant that responds only in JSON format. Responses should always be returned in JSON format with values in the specified keys. If the value for a key contains code, escape it appropriately.", # Prompt (optional) # Responses should always be in the format {\"specified_key\": \"enter response here\"}.
                            #system= "You are an assistant that responds only in JSON format. Adhere strictly to the JSON format, and when inserting code into the specified key values, include the code as a string. Also, properly escape characters that require escaping (e.g., newlines, double quotes).",
                            messages=chat_history,
                            #extra_headers = {"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                            #messages=[
                            #    {"role": "user", "content": prompt},
                            #    #{"role":"assistant", "content": "Here is the JSON requested:\n{"}
                            #]
                        )

                        response_flag = True

                    except InternalServerError as e:
                        if attempt == max_retries - 1:
                            raise  # Re-raise the exception if max retries reached
                        print(f"InternalServerError occurred. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff: double the wait time

                    if response_flag:
                        break
            
                text = message.content[0].text
                print(text)

                input_token = message.usage.input_tokens
                output_token = message.usage.output_tokens
                print(message.usage.input_tokens)
                print(message.usage.output_tokens)

                update_token(input_token, 0, token_path)

                code_blocks, error_text = extract_json_response(llm_choice, text)
                if code_blocks is not True:
                    break
                else:
                    # prompt_count = load_state()
                    write_prompt(database_dir, f"llm", code_blocks, chat_dir, count_path)
                    
                    update_token(0, output_token, token_path)
                    #chat_history.append({"role": "assistant", "content": "Too long response"})
                    #chat_history.append({"role": "user", "content": f"The answer length exceeds 4096 tokens. Please make sure it is {output_max} tokens or less and respond again. Return only JSON format data without including any text."})

                    if output_token > output_max: #4000: # Already hit the limit at 4073
                        print("Too long response")
                        chat_history.append({"role": "assistant", "content": "Too long response"})

                        print(f"Analyzing {long_count}")
                        # 20 lines (or more) didn't work here. Interesting! Haven't tested the range between 10-20 lines though.
                        addition = f"""The answer exceeds {output_max} tokens in length.
When including code in the response, even if it's in the middle of a logical unit (function, data structure, etc.), please divide the code in the JSON key into chunks of 100 lines segments and answer the first segment now. Please make sure not to truncate the JSON data in your response.
Also, if there is remaining code, set the value of the 'ongoing' key to a boolean value of True. If the code is the final part, set the value of the 'ongoing' key to a boolean value of False.
"""
                        # When splitting, set the value of the 'parsable' key in the JSON response to false.
                        #print(prompt)
                        # Responses must always be strictly in JSON format only, without any explanatory text or additional text.
                        # If a single unit that can be parsed by ctags, such as a function or data type starting from a line with no indentation, is too long, split it midway.
                        # Make sure it is {output_max} tokens or less and respond again.
                        # When splitting midway, enter the boolean value False for the 'parsable' key value.
                        # Return only JSON format data without including any text.

                        write_prompt(database_dir, f"request", addition, chat_dir, count_path)
                        # prompt_count = load_prompt_count()
                        write_prompt(database_dir, f"user", addition, chat_dir, count_path)

                        chat_history.append({"role": "user", "content": f"{addition}"})
                        #chat_history.append({"role": "user", "content": f"The answer length exceeds 4096 tokens. Please make sure it is {output_max} tokens or less and respond again. Return only JSON format data without including any text."})
                        
                        long_count += 1
                    else:
                        print("Imappropriate response format")
                        chat_history.append({"role": "assistant", "content": "Imappropriate response format"})
                        #chat_history.append({"role": "user", "content": f"The response in JSON format could not be correctly JSON decoded. If Rust code is included in the response, please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. Return only JSON format data without including any text."})
                        #chat_history.append({"role": "user", "content": f"The response in JSON format could not be decoded correctly. Please do not surround the JSON content with backticks. Respond again with only JSON data and no text."})

                        if error_text is None:
                            print("Error in addition_error1")
                            #addition_error1 = f"The response in JSON format could not be decoded correctly. Please respond again with only one JSON data and no text so that it can be processed with json.loads() and also please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. Please include the explanation text in the reason text within the JSON data. Please respond with only one pure JSON data without using code blocks or markdown syntax such as ```json <text>```."
                            addition_error1 = f"The response in JSON format could not be decoded correctly. Please wrap your JSON response in ```json ... ``` markdown code blocks. Properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. When representing backslashes as byte literals, use four backslashes. When representing backslashes as character literals, use two backslashes."
                            #addition_error1 = f"The response in JSON format could not be decoded correctly. If Rust code is included in the response, please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure.  When representing backslashes as byte literals, escape the backslash twice in the source code, and also escape it again in the byte literal, resulting in four backslashes (double backslashes). When representing backslashes as character literals, escape the backslash once in the source code and again in the character literal, resulting in two backslashes. Respond again with only JSON data and no text."
                            chat_history.append({"role": "user", "content": f"{addition_error1}"})
                            # prompt_count = load_state()
                            write_prompt(database_dir, f"user", addition_error1, chat_dir, count_path)

                        else:
                            if risky_error > 10:
                                raise ValueError("Stop due to bad format.")
                            print("Error in addition_error2")
                            #chat_history.append({"role": "user", "content": f"The response in JSON format could not be decoded correctly. Please base64 encode the rust_code in the response using encoded = base64.b64encode(text.encode('utf-8')), and return it as the value of \"rust_code\" in JSON format."})
                            #addition_error2 = f"The response in JSON format could not be decoded correctly. Please respond again with only one JSON data and no text so that it can be processed with json.loads() and also please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. Please include the explanation text in the reason text within the JSON data. Please respond with only one pure JSON data without using code blocks or markdown syntax such as ```json <text>```."
                            addition_error2 = f"The response in JSON format could not be decoded correctly. Please wrap your JSON response in ```json ... ``` markdown code blocks. Properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. Please include the explanation text in the reason field within the JSON data."
                            #addition_error2 = f"The response in JSON format could not be decoded correctly. If Rust code is not included in the response, then please respond again with only JSON data and no text and also please properly escape characters that require escaping (e.g., newlines, double quotes) while maintaining the original JSON structure. If Rust code ('rust_code' key value) is included, for the rust_code field value, please use base64-encoded Rust code (using base64.b64encode(text.encode('utf-8'))). Do not use the raw Rust code containing escape characters and line breaks."
                            #please encode the rust_code included in your response using base64.b64encode(text.encode('utf-8')), and include it as the value of 'rust_code' in JSON format."
                            chat_history.append({"role": "user", "content": f"{addition_error2}"})
                        
                            # prompt_count = load_state()
                            write_prompt(database_dir, f"user", addition_error2, chat_dir, count_path)

                            risky_error += 1


    if memory_type is not None:
        if llm_choice in ['gpt', 'gpt_azure', 'gpt_azure_databricks', 'claude', 'claude_azure', 'claude_bedrock']:
            chat_history.append({"role": "assistant", "content": text})
        
        elif llm_choice == 'Gemini':
            reverse_gemini_history(gemini_history)
        
        write_json(history_path, chat_history)

    #global ask_count
    # write_exp_data(exp_data, input_token, output_token)
    #ask_count += 1

    write_prompt(database_dir, f"response", code_blocks, chat_dir, count_path)
    # prompt_count = load_prompt_count()
    write_prompt(database_dir, f"llm", code_blocks, chat_dir, count_path)
    
    update_token(0, output_token, token_path)

    return code_blocks



def get_tool_cmd(WO_TOOL, tool_string):
    tool_cmd = f"- Please include commands to create valid input files using appropriate standard tools: {tool_string}. Do not manually construct file headers or use placeholder data. Invalid files cause early rejection and low coverage."
        
    if WO_TOOL is True:
        tool_cmd = f"- Please also include commands to create input files within the shell script."
                        
    return tool_cmd


def get_path_info(callee_main_path, target_entry):
    prompt = []
    candidates = []
    
    callee_data = read_json(callee_main_path)
    
    count = 1
    # Add a list to track path lengths
    path_lengths = []
    path_with_lengths = []  # List to store paths together with their lengths
    
    print(target_entry)
    for item in callee_data:
        if item['name'] == target_entry['target_function'] and item['file_path'] == target_entry['target_path']:
            if 'all_paths' not in item:
                path_list = []
            else:
                path_list = item['all_paths']
            for path_item in path_list:
                length = len(path_item['path'])
                # Record the path and its length
                path_with_lengths.append({
                    'path_number': count,
                    'path': path_item['path'],
                    'length': length
                })
                # Record the path length
                path_lengths.append(length)
                count += 1
            
            break
    
    # Calculate the minimum and maximum path lengths
    min_length = min(path_lengths) if path_lengths else 0
    max_length = max(path_lengths) if path_lengths else 0
    
    prompt.extend(["## Fucntion call relationship from main():", "Please write a test case that reaches that target function which follows the function calls below."])
    
    # Add only the path with the minimum length
    for path_info in path_with_lengths:
        if path_info['length'] == min_length:
            candidates.append(f"# Target path")  # {path_info['path_number']}")
            candidates.extend(path_info['path'])
            candidates.append("")
            break
    
    prompt.extend(candidates)
    
    return prompt, count, min_length, max_length


def get_path_info_wide(callee_main_path, target_entry, function_branch_path):
    prompt = []
    candidates = []
    
    callee_data = read_json(callee_main_path)
    
    count = 1
    
    # Add a list to track path lengths
    path_lengths = []
    
    all_paths = []
    print(target_entry)

    if callee_data is not None:  # added

        for item in callee_data:
            if item['name'] == target_entry['target_function'] and item['file_path'] == target_entry['target_path']:
                if 'all_paths' not in item:
                    path_list = []
                else:
                    path_list = item['all_paths']
                
                for path_item in path_list:
                    candidates.append(f"Path candidate {count}")
                    i = 1
                    for tiny_path in path_item['path']:
                        if i == 1:
                            candidates.append(f"{tiny_path}")
                        else:
                            candidates.append(f"-> {tiny_path}")
                        i += 1
                    # candidates.extend(path_item['path'])
                    length = len(path_item['path'])
                    # Record the path length
                    path_lengths.append(length)
                    candidates.append("")
                    count += 1

                    all_paths.extend(path_item['path'])
                
                break


    # Calculate the minimum and maximum path lengths
    min_length = min(path_lengths) if path_lengths else 0
    max_length = max(path_lengths) if path_lengths else 0
    
    prompt.extend(["## Fucntion call relationship from main():", "Please select one path from the possible paths below and write a test case that reaches that target function."])
    prompt.extend(candidates)
    
    ####
    all_paths = list(set(all_paths))
    result = {}
    for item in all_paths:
        result[item] = False
    ####

    cov_data = read_json(function_branch_path)
    if cov_data is not None:
        for target_item in result:
            target_name, target_path, target_line = parse_function_id(target_item)
            found = False

            for file_path, item in cov_data['files'].items():
                for func_item in item['functions']:
                    if file_path == target_path and func_item['name'] == target_name:
                        if func_item['called'] is True:
                            result[target_item] = True
                            found = True
                    if found:
                        break
                if found:
                        break
        
    covered_prompt = []
    covered_prompt.append("## Alreadly covered functions when the previously generated testcase (run_test.sh) was executed")
    covered_prompt.append("### Covered functions")
    count = 0
    for key, value in result.items():
        if value == True:
            covered_prompt.append(key)
            count += 1

    if count == 0:
        covered_prompt.append("None")

    # covered_prompt.append("") 
    # covered_prompt.append("### Uncovered path")
    # for key, value in result.items():
    #     if value == False:
    #         covered_prompt.append(key)

    return prompt, count, min_length, max_length, covered_prompt



def trim_code(target_path, file_code, given_limit, model="gpt-4"):
    if not os.path.isfile(target_path):
        return
    
    if file_code is None:
        return
    
    if not isinstance(file_code, str):
        file_code = str(file_code)
    
    encoder = tiktoken.encoding_for_model(model)
    full_tokens = len(encoder.encode(file_code))
    
    if full_tokens <= given_limit:  # If already within the limit, return the whole content
        # print("Within limit")
        return file_code
    
    # Use binary search to find the largest portion that fits within the limit
    left, right = 0, len(file_code)
    best_length = 0
    
    while left <= right:
        mid = (left + right) // 2
        # Get the candidate substring
        candidate = file_code[:mid]
        # Calculate the number of tokens in that portion
        tokens = len(encoder.encode(candidate))
        
        if tokens <= given_limit - 50:  # Leave some room for the omission message
            best_length = mid
            left = mid + 1
        else:
            right = mid - 1
    
    # Cut at the last complete line
    trimmed = file_code[:best_length]
    last_newline = trimmed.rfind('\n')
    if last_newline != -1:
        trimmed = trimmed[:last_newline + 1]
    
    # Add an omission message (based on token count)
    remaining_tokens = full_tokens - len(encoder.encode(trimmed))
    trimmed += f"\n... ( remaining_tokens is {remaining_tokens}.) Exceeding token limit, content truncated. To view the complete content of {target_path}, please use read_data mode and set file_slice (specified range) to read each section separately."
    
    print("trimmed")
    return trimmed


# Revision proposal
def trim_data(work_dir, target_path, file_code, given_limit):
    # given_limit = 8000 #10000
    model = "gpt-4"

    # # Handle the case where file_code is None or is not a string
    # if file_code is None:
    #     return

    write_file(target_path, file_code)
    
    # Convert to string type
    if not isinstance(file_code, str):
        file_code = str(file_code)
    
    encoder = tiktoken.encoding_for_model(model)
    full_tokens = len(encoder.encode(file_code))
    
    # The following is the same as the original code
    if full_tokens <= given_limit:  # If already within the limit, return the whole content
        print("Within limit")
        return file_code
    
    # Use binary search to find the largest portion that fits within the limit
    left, right = 0, len(file_code)
    best_length = 0
    
    while left <= right:
        mid = (left + right) // 2
        # Get the candidate substring
        candidate = file_code[:mid]
        # Calculate the number of tokens in that portion
        tokens = len(encoder.encode(candidate))
        
        if tokens <= given_limit - 50:  # Leave some room for the omission message
            best_length = mid
            left = mid + 1
        else:
            right = mid - 1
    
    # Cut at the last complete line
    trimmed = file_code[:best_length]
    last_newline = trimmed.rfind('\n')
    if last_newline != -1:
        trimmed = trimmed[:last_newline + 1]
    
    # Add an omission message (based on token count)
    remaining_tokens = full_tokens - len(encoder.encode(trimmed))
    trimmed += f"\n... ( remaining_tokens is {remaining_tokens}.) Exceeding token limit, content truncated.. To view the complete content of {target_path}, please use read_data mode and set file_slice (specified range) to read each section separately."
    
    print("trimmed")
    return trimmed



def get_dir_struct(app_type, target_dir, original_target_dir):  # , original_dir=None
    print(f"Getting directory structure: {target_dir}")
    
    # grant_permissions(target_dir) # off at this moment

    print("---------------")
    print(target_dir)
    print(original_target_dir)

    parent_path = None
    if original_target_dir is not None:
        parent_path = os.path.dirname(original_target_dir)
    print(parent_path)
    print("---------------")

    # List of directories to exclude
    # excluded_dirs = ['build', '.github', '.gitlab', '.tx', '.git', 'fuzz', 'test']  # ['target']
    # excluded_dirs = ['build', '.github', '.gitlab', '.tx', '.git', 'fuzz', 'test']  # ['target']
    
    # List of directories and extensions to exclude
    excluded_dirs = []
    excluded_extensions = []

    if app_type == "testcase":
        excluded_extensions = ['.gcno', '.gcda', '.o', '.obj']  # No directories
        
    elif app_type in ["translation", "s_repair"]:
        excluded_dirs = ['target']  # Directory names

    result = []
    if not os.path.exists(target_dir):
        return f"Error: Directory '{target_dir}' does not exist"
    
    def check_original_existence(path):
        if not parent_path:
            return False
        # Construct the corresponding path in original_dir
        relative_path = os.path.relpath(path, target_dir)
        original_path = os.path.join(parent_path, relative_path)
        return os.path.exists(original_path)
    
    def add_tree(dir_path, prefix=""):
        contents = sorted(Path(dir_path).iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        for i, item in enumerate(contents):
            # Skip if it is an excluded directory
            if item.is_dir() and item.name in excluded_dirs:
                continue

            if item.is_file() and any(str(item).endswith(ext) for ext in excluded_extensions):
                continue

            is_last = (i == len(contents) - 1)
            
            curr_prefix = "└── " if is_last else "├── "
            next_prefix = "    " if is_last else "│   "
            
            # Check whether it exists in the original
            exists_in_original = check_original_existence(item)
            comment = ""
            # comment = "   # Originally exists" if exists_in_original else ""
            line_num = count_file_lines(item)
            line_comment = f"  # {line_num}Lines" if line_num else ""

            result.append(f"{prefix}{curr_prefix}{item.name}{comment}{line_comment}")
            
            if item.is_dir():
                add_tree(item, prefix + next_prefix)
    
    root = Path(target_dir)
    result.append(root.name)
    
    add_tree(target_dir)
    return "\n".join(result)


def find_matching_path(workspace_dir, target_suffix):

    matching_paths = []
    matching_path = target_suffix
    for root, _, files in os.walk(workspace_dir):
        for file in files:
            full_path = os.path.join(root, file)
            if full_path.endswith(target_suffix):
                matching_paths.append(full_path)
                matching_path = full_path
                break
    
    return matching_path


def delete_lines(file_path, start_line, end_line):
    try:
        with open(file_path, 'r') as file:  # Read the file contents
            lines = file.readlines()
        
        if start_line < 1 or end_line > len(lines) or start_line > end_line:  # Check the range to delete
            print("Invalid line range.")
            return
        
        del lines[start_line - 1:end_line]  # Delete the specified lines
        
        if lines and lines[-1].endswith('\n'):  # Remove the newline of the last line (so the file does not end with a newline)
            lines[-1] = lines[-1].rstrip('\n')
         
        with open(file_path, 'w') as file:  # Write the updated contents back to the file
            file.writelines(lines)
        
        # print(f"Deleted lines from {start_line} to {end_line}.")
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except IOError:
        print(f"Error processing file '{file_path}'.")


def insert_modified_data(mod): #current_code_length

    last_count = count_file_lines(mod['file_path'])

    if mod['current_code_found'] == True:
        #print("current_code_found is True in insert_modified_data()")
        part_03_code = read_specific_lines(mod['file_path'], mod['current_end_line'] + 1, last_count) # mod['start_line'] + current_code_length

    else:
        #print("current_code_found is False in insert_modified_data()")
        part_03_code = read_specific_lines(mod['file_path'], mod['start_line'], last_count)

    #print(last_count)
    delete_lines(mod['file_path'], mod['start_line'], last_count)

    #sprint(last_count)
    #print(part_03_code)
    append_file(mod['file_path'], '\n')
    append_file(mod['file_path'], mod['modified_data'])

    append_file(mod['file_path'], '\n')
    append_file(mod['file_path'], part_03_code)


def reflect_line_modification(modifications, work_dir):
    if not isinstance(modifications, list):  # If modifications is not a list, convert it to a list
        modifications = [modifications]

    ## added
    new_modifications = []
    for item in modifications:
        if 'file_path' not in item:
            continue
        test_path = item['file_path']
        if not os.path.exists(test_path):
            test_path = find_matching_path(work_dir, test_path)
            item['file_path'] = test_path
        
        new_modifications.append(item)
    modifications = new_modifications

    ## added

    # Handle the shorthand case where end_line = -1
    for item in modifications:
        end_line = item['end_line']
        if end_line == -1:
            # print(f"Found short hand: {item}")

            # write_file(item['file_path'], item['modified_data'])
            # print(f"Created a new file or modified the entire file: {item}")

            # if item.get('is_JSON', False):
            if item.get('is_JSON', False) or isinstance(item['modified_data'], (dict, list)):
                write_file(item['file_path'], json.dumps(item['modified_data'], indent=4, ensure_ascii=False))
            else:
                write_file(item['file_path'], item['modified_data'])

    # Processing
    for item in modifications:  # Insert current_code
        # Is this okay when the file is extremely long?
        if item['end_line'] == -1:
            print(f"Finished for end_line = -1")

        else:
            # print(f"Found start_line? {item}")
            # print(f"modifications is {modifications}")

            item['current_code'] = read_specific_lines(item['file_path'], item['start_line'], item['end_line'])
            item['current_end_line'] = item['end_line']
            item['current_code_found'] = True
        
    # global reflect_count
    reflect_timestamp = get_timestamp()
    # write_json(f'look_modifs/look_modifs{str(reflect_timestamp)}.json', modifications)
    # reflect_count = reflect_count + 1

    # Group modifications by file
    file_modifications = {}
    for mod in modifications:
        if mod['file_path'] not in file_modifications:
            file_modifications[mod['file_path']] = []
        file_modifications[mod['file_path']].append(mod)

    # Apply modifications to each file
    interval_path = 'interval_mod.txt'
    for test_path, mods in file_modifications.items():

        mods.sort(key=lambda x: x['start_line'], reverse=True)  # Sort modifications in descending order of line number (to apply them from the end)

        for mod in mods:
            if mod['end_line'] == -1:
                continue

            if 'is_deletion' in mod and mod['is_deletion'] is True:
                delete_lines(mod['file_path'], mod['start_line'], mod['end_line'])
                # print(f"deleted: {mod}")
                continue

            if 'overwrite_all' in mod and mod['overwrite_all'] is True:
                # delete_file(mod['file_path'])
                write_file(mod['file_path'], mod['modified_data'])
                # print(f"overwritten: {mod}")
                continue

            offset = 0
            start_line = mod['start_line'] - 1  # 0-indexed
            # print(f"Fixing line: {mod['start_line']} in {mod['file_path']}")
            # if 'current_code' not in mod:  # This is causing the problem. Remove this for now.
                # return False  # Remove this for now
            
            if not ('current_code_found' in mod and mod['current_code_found'] == True):
                mod['current_code_found'] = False
                # print("Not found current_code")
                # print(f"{mod['modified_data']}")
                # continue

                if 'modified_data' in mod:  # To avoid errors here
                    write_file(interval_path, mod['modified_data'])
                    insert_modified_data(mod)  # , current_code_length
            else:
                # print("Found current_code")

                current_code = read_specific_lines(mod['file_path'], mod['start_line'], mod['current_end_line'])
                write_file(interval_path, current_code)
                current_code_length = count_file_lines(interval_path)

                # Calculate the difference in the number of lines between the original code and the modified code
                if 'modified_data' in mod:  # To avoid errors here
                    write_file(interval_path, mod['modified_data'])
                    modified_data_length = count_file_lines(interval_path)
                    # print(f"start_line {mod['start_line']}, modified_data_length: {modified_data_length}, current_code_length: {current_code_length}")
                    insert_modified_data(mod)  # , current_code_length

        # Update block information for the modified file
        # update_c_block(c_path, test_path, meta_dir)

    delete_file(interval_path)

    print("Code modifications completed.")

    """
    if app_type == "translation":
        # Identify c_key elements that should have existed at the modified locations
        # key_json = update_modified_keys(key_json, modified_lines)
        # List c_key elements that did not correspond to the modified locations and compute unmodified_lines.
        # All other code lines are treated as modified lines.
        return file_modifications
    
    else:
        return True
    """
    
    return file_modifications



def check_excluded(target_dir, see_path):
    abs_path = os.path.abspath(see_path)

    abs_excluded_dirs = []
    excluded_dirs = []  # added — is this correct?
    for excluded_dir in excluded_dirs:
        excluded_dir = f"{target_dir}/{excluded_dir}"
        abs_excluded_dir = os.path.abspath(excluded_dir)
        abs_excluded_dirs.append(abs_excluded_dir)
    
    print(f"abs_excluded_dirs: {abs_excluded_dirs}")

    for excluded_dir in abs_excluded_dirs:
        if abs_path.startswith(excluded_dir):
            return True
    
    # If it is not included in any excluded directory
    return False




def add_line_numbers_custom(input_file, fixed_number):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as temp_file:
            with open(input_file, 'r', encoding='utf-8') as infile:
                # Read all lines and get the maximum indentation level and line count
                lines = list(infile)
                if not lines:
                    # print(f"File {input_file} is empty.")
                    return
                max_line_num = len(lines)
                max_indent = max((len(line) - len(line.lstrip())) // 4 for line in lines)
                
                # Calculate the width for line numbers and indentation levels
                line_num_width = len(str(max_line_num))
                indent_width = len(str(max_indent))
                
                # Create format string (fix the position of :)
                format_str = f"Line{{:{line_num_width}d}} [{{:{indent_width}d}}]: {{}}"
                
                # Process each line
                for line_number, line in enumerate(lines, start=fixed_number):
                    indent_level = (len(line) - len(line.lstrip())) // 4
                    numbered_line = format_str.format(line_number, indent_level, line)
                    temp_file.write(numbered_line)
                
        # Overwrite the original file with the temporary file
        os.replace(temp_file.name, input_file)
        #print(f"Wrote file with line numbers and indent levels to {input_file}.")
    except IOError as e:
        print(f"An error occurred: {e}")


def get_lined_specific_code(work_dir, database_dir, test_path, start_line, end_line):
    if not os.path.exists(test_path):
        test_path = find_matching_path(work_dir, test_path)

    target_code = read_specific_lines(test_path, start_line, end_line)
    
    lined_test_path = f"{database_dir}/lined.txt" #"lined.c"
    write_file(lined_test_path, target_code)
    add_line_numbers_custom(lined_test_path, int(start_line)) #add_line_numbers(lined_test_path)
    test_code = read_file(lined_test_path)

    delete_file(lined_test_path)

    return test_code


def get_prompt_count(token_path):
    """Get the current number of prompts from token_path"""
    if os.path.exists(token_path):
        data = read_json(token_path)
        # data is a list in the form [{"prompt_id": 0, ...}, {"prompt_id": 1, ...}, ...]
        if isinstance(data, list) and len(data) > 0:
            # Get the prompt_id of the last entry
            return data[-1]["prompt_id"] + 1
        else:
            return 0
    else:
        return 0


def get_none_count():
    global none_count
    current_count = none_count
    none_count += 1

    return current_count


# save coverage report
def save_coverage_report(cov_target, cov_report_path, token_path, branch_coverage, line_coverage, function_coverage, id_type):
   
    if  cov_target == "function":
        average_coverage = function_coverage
    elif  cov_target == "branch":
        average_coverage = branch_coverage


   
    if os.path.exists(cov_report_path):
        data = read_json(cov_report_path)
    else:
        data = {}
    
    timestamp = get_timestamp()
    if id_type == "llm":
        prompt_id = get_prompt_count(token_path)
        prompt_id = int(prompt_id) - 1

        data[str(int(prompt_id)).zfill(4)] = str(average_coverage)
    
    elif id_type == "fuzz":
        fuzz_id = get_fuzz_count()
        fuzz_id = int(fuzz_id)
        data[f"fuzz_{str(int(fuzz_id)).zfill(4)}"] = str(average_coverage)

    else:
        # none_id = get_none_count()
        # none_id = int(none_id)
        # data[f"none_{str(int(none_id)).zfill(4)}"] = str(average_coverage)
        data[f"none_{str(timestamp).zfill(4)}"] = str(average_coverage)

    write_json(cov_report_path, data)



def is_empty_string(file_code):
    if file_code is None:
        return True
    
    if file_code.strip() == "":
        return True
        
    return False



autonomous_cli_template = f"""# In "read_data" mode
{{
    "mode" : "read_data",
    "target_files" : [path/to/file1, path/to/file2, ..., path/to/fileN], 
    "file_slices" : (if necessary, otherwise None) [
        {{
            "file_path" : (file path),
            "start_line" : (start_line of the scope),
            "end_line" : (end_line of the scope),
        }},...
    ]
    "ongoing_in_mode" : true if the "answer" response in "read_data" mode is long and will continue in subsequent responses. false otherwise,
    "ongoing" : true if the response will continue in a different mode. false otherwise,
    "reason" : explanatory text for the response (insert here if needed)
}}

# In "modify_data" mode
{{
    "mode" : "modify_data",
    "answer" : [
        {{
            "file_path" : (file path),
            "start_line" : (start line of the original code to be deleted; must reflect the original range to be replaced),
            "end_line" : (end line of the original code to be deleted; must reflect the original range to be replaced),
            "is_deletion" : True for deletion only, False for modification,
            "overwrite_all" : Flag for full file modification. If true, overwrites the whole file; if false, modifies only the specified lines
            "modified_data" : (Content of the corrected code without any omission. Content of the corrected code as a string if is_JSON is false, or as a direct JSON object if is_JSON is true.),
        }},
        {{
            "file_path" : (file path),
            "start_line" : (start line of the original code to be deleted; must reflect the original range to be replaced),
            "end_line" : (end line of the original code to be deleted; must reflect the original range to be replaced),
            "is_deletion" : True for deletion only, False for modification,
            "overwrite_all" : Flag for full file modification. If true, overwrites the whole file; if false, modifies only the specified lines
            "modified_data" : (Content of the corrected code without any omission. Content of the corrected code as a string if is_JSON is false, or as a direct JSON object if is_JSON is true.),
        }},...
    ],
    "ongoing_in_mode" : true if the "answer" response in "modify_data" mode is long and will continue in subsequent responses. false otherwise,
    "ongoing" : true if the response will continue in a different mode. false otherwise,
    "ready_to_execute" : true if this response marks the end of a coherent modification set and it's ready to be tested; false otherwise,
    "max_counter":  highest file counter number used in your response. If input files are not used, use null."
    "reason" : explanatory text for the response (insert here if needed)
}}

# In "execute_command" mode
{{
    "mode" : "execute_command",
    "answer" : shell script content to be executed,
    "ongoing_in_mode" : true if the "answer" response in "execute_command" mode is long and will continue in subsequent responses. false otherwise,
    "ongoing" : true if the response will continue in a different mode. false otherwise,
    "reason" : explanatory text for the response (insert here if needed)
}}
"""

autonomous_template = f"""# In "read_data" mode
{{
    "mode" : "read_data",
    "target_files" : [path/to/file1, path/to/file2, ..., path/to/fileN], 
    "file_slices" : (if necessary, otherwise None) [
        {{
            "file_path" : (file path),
            "start_line" : (start_line of the scope),
            "end_line" : (end_line of the scope),
        }},...
    ]
    "ongoing_in_mode" : true if the "answer" response in "read_data" mode is long and will continue in subsequent responses. false otherwise,
    "ongoing" : true if the response will continue in a different mode. false otherwise,
    "reason" : explanatory text for the response (insert here if needed)
}}

# In "modify_data" mode
{{
    "mode" : "modify_data",
    "answer" : [
        {{
            "file_path" : (file path),
            "start_line" : (start line of the original code to be deleted; must reflect the original range to be replaced),
            "end_line" : (end line of the original code to be deleted; must reflect the original range to be replaced),
            "is_deletion" : True for deletion only, False for modification,
            "overwrite_all" : Flag for full file modification. If true, overwrites the whole file; if false, modifies only the specified lines
            "modified_data" : (Content of the corrected code without any omission. Content of the corrected code as a string if is_JSON is false, or as a direct JSON object if is_JSON is true.),
        }},
        {{
            "file_path" : (file path),
            "start_line" : (start line of the original code to be deleted; must reflect the original range to be replaced),
            "end_line" : (end line of the original code to be deleted; must reflect the original range to be replaced),
            "is_deletion" : True for deletion only, False for modification,
            "overwrite_all" : Flag for full file modification. If true, overwrites the whole file; if false, modifies only the specified lines
            "modified_data" : (Content of the corrected code without any omission. Content of the corrected code as a string if is_JSON is false, or as a direct JSON object if is_JSON is true.),
        }},...
    ],
    "ongoing_in_mode" : true if the "answer" response in "modify_data" mode is long and will continue in subsequent responses. false otherwise,
    "ongoing" : true if the response will continue in a different mode. false otherwise,
    "ready_to_execute" : true if this response marks the end of a coherent modification set and it's ready to be tested; false otherwise,
    "reason" : explanatory text for the response (insert here if needed)
}}

# In "execute_command" mode
{{
    "mode" : "execute_command",
    "answer" : shell script content to be executed,
    "ongoing_in_mode" : true if the "answer" response in "execute_command" mode is long and will continue in subsequent responses. false otherwise,
    "ongoing" : true if the response will continue in a different mode. false otherwise,
    "reason" : explanatory text for the response (insert here if needed)
}}
"""

def get_annotated_source_code_range(target_file_path, start_line=None, end_line=None):
    """
    Generate annotated source code from an lcov .info file
    """
    try:
        if not os.path.exists(target_file_path):
            return f"Error: File not found: {target_file_path}"
        
        # Run lcov to generate the .info output
        target_dir = os.path.dirname(target_file_path)
        
        cmd = ['lcov', '--capture', '--directory', target_dir, '--output-file', '-']
        
        result = subprocess.run(
            cmd,
            # capture_output=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=None  # 600
        )
        
        if result.returncode != 0:
            return f"lcov error: {result.stderr}"
        
        # Parse the .info output
        lines = result.stdout.split('\n')
        line_data = {}
        branch_data = {}
        function_data = {}
        current_file = None
        
        for line in lines:
            line = line.strip()
            
            # File information
            if line.startswith('SF:'):
                current_file = line[3:]
                
            elif current_file == target_file_path:
                # Line coverage: DA:line_number,execution_count
                if line.startswith('DA:'):
                    parts = line[3:].split(',')
                    line_num = int(parts[0])
                    count = int(parts[1])
                    line_data[line_num] = count
                
                # Function coverage: FN:line_number,function_name
                elif line.startswith('FN:'):
                    parts = line[3:].split(',', 1)
                    line_num = int(parts[0])
                    func_name = parts[1]
                    function_data[line_num] = func_name
                
                # Branch coverage: BRDA:line_number,block,branch,execution_count
                elif line.startswith('BRDA:'):
                    parts = line[5:].split(',')
                    line_num = int(parts[0])
                    count = 0 if parts[3] == '-' else int(parts[3])
                    if line_num not in branch_data:
                        branch_data[line_num] = []
                    branch_data[line_num].append(count)
        
        # Read the original source file
        with open(target_file_path, 'r', encoding='utf-8') as f:
            source_lines = f.readlines()
        
        # Generate annotated code
        annotated_lines = []
        
        # Add metadata header
        annotated_lines.append(f"        -:    0:Source:{os.path.basename(target_file_path)}")
        annotated_lines.append(f"        -:    0:Data:lcov-generated")
        
        for i, source_line in enumerate(source_lines, 1):
            # Check line range
            if start_line is not None and i < start_line:
                continue
            if end_line is not None and i > end_line:
                continue
                
            # Generate prefix based on coverage information
            if i in line_data:
                count = line_data[i]
                if count == 0:
                    prefix = f"    #####:{i:5}:"
                else:
                    prefix = f"{count:9}:{i:5}:"
            else:
                prefix = f"        -:{i:5}:"
            
            annotated_lines.append(f"{prefix}{source_line.rstrip()}")
            
            # Add function information
            if i in function_data:
                func_name = function_data[i]
                func_count = line_data.get(i, 0)
                annotated_lines.append(
                    f"function {func_name} called {func_count} returned 100% blocks executed 100%"
                )
            
            # Add branch information
            if i in branch_data:
                for j, branch_count in enumerate(branch_data[i]):
                    if branch_count > 0:
                        annotated_lines.append(f"branch  {j} taken 100%")
                    else:
                        annotated_lines.append(f"branch  {j} never executed")
        
        return '\n'.join(annotated_lines)
        
    except Exception as e:
        return f"Error: {str(e)}"


########################################
##### Translation
########################################

# List c_key elements that were not identified as modified and compute unmodified_lines.
# All other code lines are considered modified lines.
def get_modified_rust_lines(modified_c_keys, c_rust_path, meta_dir):
    """
    Get the corresponding Rust code line ranges from modified C keys
    
    Args:
        modified_c_keys: Set of modified C keys
        c_rust_map: Mapping dictionary from C keys to Rust keys
                   {"c_key": "rust_key"}
        meta_dir: Path to the metadata directory
    
    Returns:
        dict: Modified Rust files and their line ranges
              {file_path: {'start_line': int, 'end_line': int, 'lines': set}}
    """
    print("get_modified_rust_lines")
    c_rust_map = read_json(c_rust_path)
    modified_rust_lines = {}
    
    # Process each key in modified_c_keys
    for c_key in modified_c_keys:
        # Get the corresponding Rust key from c_rust_map
        rust_key = c_rust_map.get(c_key)
        
        if rust_key is None:
            print(f"Warning: No rust mapping found for c_key: {c_key}")
            continue
        
        # Split rust_key: "name:file_path:start:end"
        parts = rust_key.split(':')
        if len(parts) >= 4:
            rust_name = parts[0]
            rust_file_path = parts[1]
            rust_start = int(parts[2])
            rust_end = int(parts[3])
            
            # Aggregate line ranges per file
            if rust_file_path not in modified_rust_lines:
                modified_rust_lines[rust_file_path] = {
                    'start_line': rust_start,
                    'end_line': rust_end,
                    'lines': set(range(rust_start, rust_end + 1))
                }
            else:
                # Update existing entry (expand range)
                current = modified_rust_lines[rust_file_path]
                current['start_line'] = min(current['start_line'], rust_start)
                current['end_line'] = max(current['end_line'], rust_end)
                current['lines'].update(range(rust_start, rust_end + 1))
    
    # Format aggregated line ranges into final output
    result = {}
    for file_path, data in modified_rust_lines.items():
        result[file_path] = {
            'start_line': data['start_line'],
            'end_line': data['end_line'],
            'modified_lines': sorted(list(data['lines']))
        }
        print(f"  {file_path}: lines {data['start_line']}-{data['end_line']} ({len(data['lines'])} lines)")
    
    print(f"Found {len(result)} modified Rust files")
    return result



def get_grouped_c_keys(modified_c_keys, count) -> List[str]:
    """
    Split a set of keys into groups of size `count` and return a list of JSON strings
    
    Args:
        modified_c_keys: Set of keys (e.g., {"CONFIG_H:trans_c/mini/config.h:3:3", ...})
        count: Number of keys per group
        
    Returns:
        List of JSON strings containing grouped keys
    """
    print("Getting get_grouped_c_keys")
    
    # Sort by start_line
    keys_list = sorted(
        list(modified_c_keys),
        key=lambda k: int(k.split(':')[2]) if len(k.split(':')) >= 3 and k.split(':')[2].isdigit() else 0
    )

    # Split into chunks of size `count`
    grouped_jsons = []
    for i in range(0, len(keys_list), count):
        chunk = keys_list[i:i + count]
        
        # Build dictionary list
        chunk_list = []
        for key in chunk:
            # Key format: "NAME:path:start:end"
            parts = key.split(':')
            if len(parts) >= 4:
                name = parts[0]
                file_path = parts[1]
                start_line = parts[2]
                end_line = parts[3]
                
                chunk_list.append({
                    "name": name,
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line
                })
        
        # Convert to JSON string
        json_str = json.dumps(chunk_list, indent=4, ensure_ascii=False)
        grouped_jsons.append(json_str)
    
    print(f"Created {len(grouped_jsons)} groups from {len(keys_list)} keys")
    
    return grouped_jsons


def merge_with_initial(one_unit, modified_c_keys):
    """
    Add data from one_unit to modified_c_keys (without duplication)
    
    Args:
        one_unit: List of data 
                  [{"name": ..., "file_path": ..., "start_line": ..., "end_line": ...}, ...]
        modified_c_keys: Set of keys (e.g., {"NAME:path:start:end", ...})
        
    Returns:
        Merged set of keys
    """
    print("Merge with initial code items")
    
    for item in one_unit:
        name = item.get('name', '')
        file_path = item.get('file_path', '')
        start_line = item.get('start_line', '')
        end_line = item.get('end_line', '')
        
        key = f"{name}:{file_path}:{start_line}:{end_line}"
        modified_c_keys.add(key)  # Set automatically removes duplicates
    
    print(f"Added {len(one_unit)} items, total keys: {len(modified_c_keys)}")
    
    return modified_c_keys


ask_map_template = f"""# In "read_data" mode
{{
    "mode" : "read_data",
    "target_files" : [path/to/file1, path/to/file2, ..., path/to/fileN], 
    "file_slices" : (if necessary, otherwise None) [
        {{
            "file_path" : (file path),
            "start_line" : (start_line of the scope),
            "end_line" : (end_line of the scope),
        }},...
    ]
    "ongoing_in_mode" : true if the "answer" response in "read_data" mode is long and will continue in subsequent responses. false otherwise,
    "ongoing" : true if the response will continue in a different mode. false otherwise,
    "reason" : explanatory text for the response (insert here if needed)
}}

# In "modify_data" mode
{{
    "mode" : "modify_data",
    "answer" : [
        {{
            "file_path" : (file path),
            "start_line" : (start line of the original code to be deleted; must reflect the original range to be replaced),
            "end_line" : (end line of the original code to be deleted; must reflect the original range to be replaced),
            "is_deletion" : True for deletion only, False for modification,
            "no_simplification" : true if all original intended features are fully preserved, without any omissions and simplifications and placeholders. false otherwise,
            "is_JSON" :If the file_path is a JSON file, then True, otherwise False,
            "modified_data" : (Content of the corrected code without any omission. Content of the corrected code as a string if is_JSON is false, or as a direct JSON object if is_JSON is true.),
        }},
        {{
            "file_path" : (file path),
            "start_line" : (start line of the original code to be deleted; must reflect the original range to be replaced),
            "end_line" : (end line of the original code to be deleted; must reflect the original range to be replaced),
            "is_deletion" : True for deletion only, False for modification,
            "no_simplification" : true if all original intended features are fully preserved, without any omissions and simplifications and placeholders. false otherwise,
            "is_JSON" :If the file_path is a JSON file, then True, otherwise False,
            "modified_data" : (Content of the corrected code without any omission. Content of the corrected code as a string if is_JSON is false, or as a direct JSON object if is_JSON is true.),
        }},...
    ],
    "ongoing_in_mode" : true if the "answer" response in "modify_data" mode is long and will continue in subsequent responses. false otherwise,
    "ongoing" : true if the response will continue in a different mode. false otherwise,
    "reason" : explanatory text for the response (insert here if needed)
}}

# In "execute_command" mode
{{
    "mode" : "execute_command",
    "answer" : shell script content to be executed,
    "ongoing_in_mode" : true if the "answer" response in "execute_command" mode is long and will continue in subsequent responses. false otherwise,
    "ongoing" : true if the response will continue in a different mode. false otherwise,
    "reason" : explanatory text for the response (insert here if needed)
}}
"""

# "function", "others", "conditional"
# Also wondering whether this JSON format is appropriate for passing data.
# Reducing it would decrease the size, but over-reduction might lose important information.
def create_tmp_rust_json(c_path, meta_dir, label):  # , div_start_line
    c_data = obtain_metadata(c_path, meta_dir, False, False, "def")

    # In Rust, except for redefined conditional blocks, elements can be handled independently
    """
    if div_start_line is None:
        div_start_line = 1
    else:
        div_start_line = 0
    """

    new_data = []
    for item in c_data:
        if label == "function":
            if item['block_type'] == "function":
                new_json = {
                    # "block_type": item['block_type'],
                    # "element_id": item['element_id'],
                    "name": item['name'],
                    "start_line": item['start_line'],  # - div_start_line + 1,
                    # "end_line": item['end_line'] - div_start_line + 1, # some may not have end_line
                    "rust_code": None,  # to be filled by LLM
                    # "rust_start_line": None, # to be filled by LLM
                }
                new_data.append(new_json)

            if 'components' in item:
                for elem in item['components']:
                    if elem['category'] == 'function':
                        if 'end_line' not in elem:
                            elem['end_line'] = elem['start_line']

                        new_json = {
                            # "category": elem['category'],
                            # "block_type": 'function',
                            # "element_id": elem['element_id'],
                            "name": elem['name'],
                            "start_line": elem['start_line'],
                            # "end_line": elem['end_line'],
                            "rust_code": None,
                            # "rust_start_line": None,
                        }

                        new_data.append(new_json)  # flattening structure here

        elif label == "others":
            if item['block_type'] == "others":
                new_json = {
                    "name": item['name'],
                    "start_line": item['start_line'],
                    "end_line": item['end_line'],
                    "rust_code": None,
                }
                new_data.append(new_json)

            if 'components' in item:
                for elem in item['components']:
                    if elem['category'] in ['macro_func', 'macro_var', 'data_type', 'global_var']:
                        if 'end_line' not in elem:
                            elem['end_line'] = elem['start_line']
                        new_json = {
                            "name": elem['name'],
                            "start_line": elem['start_line'],
                            "end_line": elem['end_line'],
                            "rust_code": None,
                        }
                        new_data.append(new_json)

        elif label == "conditional":
            if item['block_type'] == "conditional":
                new_json = {
                    "name": item['name'],
                    "start_line": item['start_line'],
                    "end_line": item['end_line'],
                    "rust_code": None,
                }
                new_data.append(new_json)

            if 'components' in item:
                for elem in item['components']:
                    if 'block_type' in elem and elem['block_type'] == "conditional":
                        if 'end_line' not in elem:
                            elem['end_line'] = elem['start_line']

                        new_json = {
                            "name": elem['name'],
                            "start_line": elem['start_line'],
                            "end_line": elem['end_line'],
                            "rust_code": None,
                        }

                        new_data.append(new_json)  # flattening structure here

    tmp_rust_path = f"{database_dir}/rust_tmp.json"
    write_json(tmp_rust_path, new_data)  # write to rust_path

    return tmp_rust_path


def get_remaining_list(tmp_json_data, sum_modified_list, database_dir):
    # sum_modified_list = sum_modified_data["modified_data"]

    write_json(f"{database_dir}/part1.json", tmp_json_data)
    write_json(f"{database_dir}/part2.json", sum_modified_list)

    new_list = []
    if isinstance(sum_modified_list, dict):
        print("This is a dictionary.")
        sum_modified_list = sum_modified_list["modified_data"]
    elif isinstance(sum_modified_list, list):
        # If it is a list
        print("This is not a dictionary.")
        found = False
        for dict_item in sum_modified_list:
            if "modified_data" in sum_modified_list:
                found = True
                new_list.append(sum_modified_list["modified_data"])

        if found is True:
            sum_modified_list = new_list

    write_json(f"{database_dir}/part3.json", sum_modified_list)

    remain_list = []
    for item in tmp_json_data:
        found = False
        for mod_item in sum_modified_list:
            if 'name' in item and 'name' in mod_item and item['name'] == mod_item['name']:
                found = True
            if 'c_name' in item and 'c_name' in mod_item and item['c_name'] == mod_item['c_name']:
                found = True
            if 'name' in item and 'c_name' in mod_item and item['name'] == mod_item['c_name']:
                found = True
            if 'c_name' in item and 'name' in mod_item and item['c_name'] == mod_item['name']:
                found = True

            if found is True:
                break

        if not found:
            remain_list.append(item)

    return remain_list



def ask_correspondence(repair_target, interface): # repair_target, target_dir, entry, original_run_path, original_execute_path, meta_dir, dep_json_path, exp_data, repair_count # div_start_line, 

    ask_count = 1

    modified_lines = interface.modified_lines
    key_json = interface.key_json
    tmp_json_data = key_json #interface.tmp_json_data
    
    
    build_path = interface.build_path
    rust_build_path = interface.rust_build_path
    run_path = interface.rust_build_path  #f"{work_dir}/build_rust.sh" # f"{work_dir}/build_rust.sh" #interface.run_path']
    
    run_test_path = interface.run_test_path
    run_all_path = interface.run_all_path  #f"{work_dir}/build_rust.sh" # f"{work_dir}/build_rust.sh" #interface.run_path
    
    rust_path = interface.rust_path
    #c_path = interface.c_path
    one_unit = interface.one_unit
    rust_output_dir = interface.rust_output_dir
    repair_count = interface.repair_count
    rust_edition = interface.rust_edition

    work_dir = interface.work_dir
    database_dir = interface.database_dir
    #repair_target = interface.repair_target
    target_dir = interface.target_dir
    original_dir = interface.original_target_dir

    raw_dir = interface.raw_dir
    REPAIR_MAX = interface.repair_max
    

    llm_interface = interface.llm_interface
    output_max = llm_interface.output_max
    
    tmp_rust_path = f"{database_dir}/rust_tmp.json"

    if not os.path.exists(run_path):
        create_file(run_path)

    execute_path = f"{work_dir}/execute.sh" #get_execute_path(run_path) #interface.execute_path']
    if not os.path.exists(execute_path):
        create_file(execute_path)
    else:
        delete_file(execute_path)  # clear the file
        create_permissioned_file(execute_path)


    execute_dir = os.path.dirname(os.path.normpath(execute_path))

    repair_count = interface.repair_count
    #repair_target = interface.repair_target
    target_dir = interface.target_dir
    raw_dir = interface.raw_dir

    if repair_target == "build":
        # From build
        entry = interface.entry
        meta_dir = interface.meta_dir
        dep_json_path = interface.dep_json_path
        exp_data = interface.exp_data
        rust_path = interface.rust_path
        lib_path = interface.lib_path #f"{rust_output_dir}/src/lib.rs" #interface.lib_path']
        

    elif repair_target == "ask_generates":
        answer_path = interface.answer_path   #answer_path = f"{work_dir}/answer.json"
        rust_path = interface.rust_path
        meta_dir = interface.meta_dir
        dep_json_path = interface.dep_json_path
        exp_data = interface.exp_data
        #modified_files = interface.modified_files
        conds_data = read_json(interface.conds_status_path)
        lib_path = interface.lib_path
        build_path = interface.build_path
        macro_type = interface.macro_type
        target_path = interface.target_path


    elif repair_target == "ask_correspondence" or repair_target == "ask_unimplemented":
        c_path = interface.c_path
        meta_dir = interface.meta_dir
        answer_path = interface.answer_path   #answer_path = f"{work_dir}/answer.json"
        rust_path = interface.rust_path
        meta_dir = interface.meta_dir
        dep_json_path = interface.dep_json_path
        exp_data = interface.exp_data
        #modified_files = interface.modified_files

    elif repair_target == "judge_conds":
        answer_path = interface.answer_path  #answer_path = f"{work_dir}/answer.json"
        rust_path = interface.rust_path
        meta_dir = interface.meta_dir
        dep_json_path = interface.dep_json_path
        exp_data = interface.exp_data
        #modified_files = interface.modified_files
        conds_data = read_json(interface.conds_status_path)
        lib_path = interface.lib_path
        build_path = interface.build_path
        #macro_type = interface.macro_type']

    elif repair_target == "judge_macros":
        answer_path = interface.answer_path   #answer_path = f"{work_dir}/answer.json"
        rust_path = interface.rust_path
        meta_dir = interface.meta_dir
        dep_json_path = interface.dep_json_path
        exp_data = interface.exp_data
        #modified_files = interface.modified_files']
        conds_data = read_json(interface.conds_status_path)
        lib_path = interface.lib_path
        build_path = interface.build_path
        #macro_type = interface.macro_type

    # start iteration
    mode = None
    execute_error = None
    execute_out = None
    read_prompt = None

    error = True # Assume errors exist by default
    ongoing_flag = False
    mode = None

    editied_files = []

    iteration_dict = {}
    judge_dict = {}
    persable_units = {}
    module_list = []

    while (1):
        if repair_count == REPAIR_MAX: # exp_data['repair_count'] == REPAIR_MAX:
            print(f"Force to finish. Hit the REPAIR_MAX ({REPAIR_MAX}).")
            iteration_dict[rust_path] = repair_count
            sys.exit(1)  #return True
        
        if mode != "read_data":
            if (repair_count != 1 and repair_target == "build") or repair_target == "compile": #if (repair_count != 1 and (repair_target == "build" or repair_target == "compile")): # This shouldn't be repair_count != 1, right?
                error, std_out, repair_count = run_script(run_path, 50, True, None, "both", None, repair_count, None, None, mode)
                print(f"Judging at run_script: error: {error}")

        # exp_data['repair_count'] = repair_count
        # exp_data['phase'] = 'repair'
    
        # if ongoing_flag is False and error is None and mode != "read_data": # This condition is tricky because ongoing_flag might not be functioning correctly.
        #     break

        if repair_target in ["ask_generates", "ask_correspondence", "ask_unimplemented", "judge_conds",  "judge_macros"]:
            if repair_count != 1 and mode != "read_data" and ongoing_flag is False:
                break

        print(f"Judging at {repair_count}: mode: {mode}, ongoing_flag: {ongoing_flag}, error: {error}")
        if error is None and mode != "read_data" and ongoing_flag is False:
            break

        
        if repair_target == "build":

            conds_status_path = interface.conds_status_path
            conds_data = read_json(conds_status_path)

            macro_list = []
            for macro, item in conds_data.items():
                if item['defined'] is True:
                    macro_list.append({
                        "name": macro,
                        "value": item['value']
                    })

            if repair_count == 1:

                prompt = [f"The following macro variables are those defined (enabled) during C compilation. Please convert their equivalents to Rust and write them to {lib_path}.",
                        #"For each corresponding item, please insert it at the necessary file location.",
                        "When converting, please follow the translation instructions below and respond in JSON format.",
                        "When responding, please follow the response rules below and select only one of the following three modes to generate your response.",
                        ]

                sentence = []
                if macro_list:
                    prompt.extend(["", "## Defined macro variables and their values:"])
                    for item in macro_list:
                        sentence.append(f"{item['name']}: {item['value']}")
                
                prompt.extend(sentence)


                prompt.extend(["", "## Translation instructions:",
                               "- The Rust program will be generated as a library crate.",
                               #"- Please create the project using the --lib option.",
                               "- The target code will be compiled on Linux (Ubuntu 22.04 LTS).",
                               f"- Use only {lib_path}, and do not create src/main.rs.",
                               "- Since other elements will be converted step by step afterwards, please convert only the specified code elements and do not add extra function definitions.",
                               #"- In Cargo.toml, use the [lib] section and specify crate-type = [\"cdylib\", \"rlib\"].",
                ])
            
            else:
                #rust_build_path = f"build_rust.sh" #f"{}"
                if error is None:
                    prompt = [f"Please continue providing the answer on how to resolve the errors that occur when compiling {rust_output_dir}."]
                    #prompt = [f"Please continue providing the answer on how to resolve the errors that occur when executing {rust_build_path}."]

                else:
                    if ongoing_flag is False:
                        prompt = [f"The following errors occurred when compiling {rust_output_dir}. Please tell me how to resolve them."]

                    else:
                        prompt = [f"Please continue providing the answer on how to resolve the errors that occur when compiling {rust_output_dir}."]

                prompt.extend(["When responding, please select only one of the three modes to generate your response."])


        if repair_target == "ask_generates":
            conds_status_path = interface.conds_status_path
            conds_data = read_json(conds_status_path)

            macro_list = []
            for macro, macro_info in conds_data.items():
                if macro_info['macro_type'] == macro_type:
                    macro_list.append({
                        "name": macro,
                    })
                """
                if 'usages' in macro_info and len(macro_info['usages']) > 1: # There are some that don't have this for some reason, which is suspicious...
                    item = macro_info['usages'][0] #: # Must use usages, otherwise items without definitions won't be captured. #if item['defined'] is True:
                    if macro_info['macro_type'] == macro_type:
                        macro_list.append({
                            "name": macro,
                            #"value": item['value']
                        })
                """

            if repair_count == 1:
                prompt = []
                prompt.extend([f"The following C macro variables have been translated into Rust and written into {target_path}.",
                            "The original C source code retains the JSON-formatted metadata below for elements related to macro variables named 'name'.",
                            #"Cross-reference the C code before translation with the translated Rust code, add two new key: 'rust_start_line' indicating where the Rust logical block including the corresponding Rust code begins, and 'rust_end_line' indicating where it ends.",
                            f"Refer to the following translated Rust code in {target_path} and update each element block in the original C JSON metadata by adding two new key: 'rust_start_line' indicating where the corresponding Rust code begins, and 'rust_end_line' indicating where it ends.", #a new key 'rust_code' and filling in the corresponding translated Rust code.",
                            "When responding, follow the response rules below and choose only one of the three modes to generate your response."
                ])
                
                prompt.extend(["", "## Response rules:",
                              f"- Please provide your answer by writing the JSON content of the {answer_path} as 'modified_data' in 'modify_data' mode.",
                              f"- For file_path, write a relative path in the format '{work_dir}/path/to/file'", #f"- For file_path, write a relative path starting from {work_dir}.",
                              f"- Since {answer_path} will continue to be updated, set 'start_line' to 1 and 'end_line' to -1 for the {answer_path} modification area.",
                              "- When writing the translated code in the 'rust_code' field of the JSON response, please ensure that line breaks and indentation are properly reflected.",
                ])
                            
                prompt.extend(["\n## Response format", "In summary, please respond in the following JSON format:"]) 
                #prompt.extend([json_template])
                #prompt.extend([autonomous_template])
                prompt.extend([ask_map_template])


                sentence = []
                if macro_list:
                    prompt.extend(["", "## Macro variables defined in the original C program:"])
                    for item in macro_list:
                        prompt.append(f"{item['name']}")  # {item['value']}

                rust_code = get_lined_code(target_path, work_dir)
                prompt.extend(["", f"## Rust Code in {target_path}", rust_code])
                
                tmp_json_data = []
                if macro_list:
                    for item in macro_list:
                        tmp_json_data.append({
                            "name" : item['name'], 
                            "rust_start_line" : None,
                            "rust_end_line" : None,
                            #"rust_code" : None,
                        })
                write_json("tmp_macro.json", tmp_json_data)
                tmp_json_data = read_file("tmp_macro.json")
                delete_file("tmp_macro.json")

                # for macro, item in conds_data.items():
                #     tmp_json_data.append({
                #         "name" : macro, 
                #         "rust_code" : None,
                #         "file_path" : None,
                #         "start_line" : None,
                #         "end_line" : None,
                #     })

                prompt.extend(["", "## The original C JSON-formatted metadata:", tmp_json_data])
            
            else:  # if ongoing_flag is False:
                prompt = ["Please continue providing JSON data responses with corresponding Rust code for each C macro variable."]

                prompt.extend(["", "## Response rules:",
                              f"- Please provide your answer by writing the JSON content of the {answer_path} as 'modified_data' in 'modify_data' mode.",
                              f"- Since {answer_path} will continue to be updated, set 'start_line' to 1 and 'end_line' to -1 for the {answer_path} modification area.",
                              "- When writing the translated code in the 'rust_code' field of the JSON response, please ensure that line breaks and indentation are properly reflected.",
                ])

                prompt.extend(["\n## Response format", "In summary, please respond in the following JSON format:"]) 
                #prompt.extend([json_template])
                #prompt.extend([autonomous_template])
                prompt.extend([ask_map_template])

        if repair_target == "ask_correspondence":
            label = interface.label

            if repair_count == 1:

                prompt = []
                prompt.extend(["Now please cross-reference the C code before translation with the Rust code after translation, add a new key 'rust_code' to the metadata elements of each block in the JSON format of the original C code, and enter the translated code for each element.", #"The following Rust code has been translated from the original C source code and compiled successfully.",
                                "The original C source code retains the following JSON-formatted metadata related to the function named 'name', enclosed by the start line (start_line) and end line (end_line).",
                                #"Now please cross-reference the C code before translation with the Rust code after translation, and add a new key 'rust_code' to each function entry in the JSON metadata.",
                                "When responding, please follow the response rules below and generate your response by selecting only one of the following three modes.",
                                # "For the 'rust_start_line' key, write the line number in the Rust code where each 'rust_code' appears in the following Rust code after translation.",
                                #"Provide the response in the value of the 'answer' key in JSON format.",
                                ])

                prompt.extend(["", "## Response rules:", 
                                f"- Do NOT use read_data mode to read {answer_path}. It is the output file and will be populated by your modify_data responses.",
                                "- Please provide responses only for the Rust code equivalents of the JSON metadata shown below. Do not respond with code elements other than the JSON metadata shown below.",
                                "- Please change the value of the 'rust_code' key according to the category of the element:",
                                f"    - Functions: Please write only the function signature for each function, excluding the function body entirely, ensuring that input and output types are visible.",
                                f"    - Others: Please determine the line numbers (rust_start_line and rust_end_line) based on logical Rust code blocks rather than direct line-by-line correspondence. A block should represent a complete logical unit such as an entire struct definition, conditional block and so on.",# This ensures the translated code maintains its semantic structure rather than being split mid-block.",
                                #"- Please determine the line numbers (rust_start_line and rust_end_line) based on logical Rust code blocks rather than direct line-by-line correspondence. A block should represent a complete logical unit such as an entire struct definition, conditional block and so on.",# This ensures the translated code maintains its semantic structure rather than being split mid-block.",
                                #"- For variables and fields that are part of a struct, please provide the line numbers for the entire struct definition block that contains them, not just the individual line numbers.",
                                #"- As the value for the 'rust_code' key, write only the function signature for each function or macro function, excluding the function body entirely, ensuring that input and output types are visible.",
                                #"- In addition, as the value for the 'rust_function_name' key, write only the function name.",
                                "- If there is no corresponding item, set the \"have_correspondence\" flag to False. If there is, set the \"have_correspondence\" flag to True.",
                                "- If the have_correspondence flag is False, write a code comment in the no_equivalent_reason field explaining why an equivalent Rust implementation does not exist.",
                                f"- Please provide your answer by writing the JSON content of the {answer_path} as 'modified_data' in 'modify_data' mode.",
                                f"- Since {answer_path} will continue to be updated, set 'start_line' to 1 and 'end_line' to -1 for the {answer_path} modification area.",
                                "- When writing the translated code in the 'rust_code' field of the JSON response, please ensure that line breaks and indentation are properly reflected.",
                    ])

                """
                # depreciated
                if label == 'conditional': # 'conditional'
                    prompt.extend(["Now please cross-reference the C code before translation with the Rust code after translation, add a new key 'rust_code' to the metadata elements of each block in the JSON format of the original C code, and enter the translated code for each element.", #"The following Rust code has been translated from the original C source code and compiled successfully.",
                                    "The original C source code retains the following JSON-formatted metadata related to the block enclosed by the start line (c_start_line) and end line (c_end_line).",
                                    #"Now please cross-reference the C code before translation with the translated Rust code, add two new key to all of the elements (which are identified by element_id) respectively: 'rust_start_line' indicating where the Rust logical block including the corresponding Rust code begins, and 'rust_end_line' indicating where it ends.",
                                    #"Cross-reference the C code before translation with the Rust code after translation, add a new key 'rust_code' to the metadata elements of each block in the JSON format of the original C code, and enter the translated code for each element.",
                                    "When responding, please follow the response rules below and generate your response by selecting only one of the following three modes.",
                                    #"Please provide responses only for the Rust code equivalents of the JSON metadata shown below. Do not respond with code elements other than the JSON metadata shown below.",
                                    ])

                    prompt.extend(["", "## Response rules:",
                                "- Please determine the line numbers (rust_start_line and rust_end_line) based on logical Rust code blocks rather than direct line-by-line correspondence. A block should represent a complete logical unit such as an entire struct definition, conditional block and so on.",# This ensures the translated code maintains its semantic structure rather than being split mid-block.",
                                "- Please provide responses only for the Rust code equivalents of the JSON metadata shown below. Do not respond with code elements other than the JSON metadata shown below.",
                                "- If there is no corresponding item, set the \"have_correspondence\" flag to False (and set rust_name, rust_start_line, and rust_end_line to None). If there is, set the \"have_correspondence\" flag to True."
                                "- If the have_correspondence flag is False, write a code comment in the no_equivalent_reason field explaining why an equivalent Rust implementation does not exist.",
                                f"- Please provide your answer by writing the JSON content of the {answer_path} as 'modified_data' in 'modify_data' mode.",
                                f"- Since {answer_path} will continue to be updated, set 'start_line' to 1 and 'end_line' to -1 for the {answer_path} modification area.",
                                "- When writing the translated code in the 'rust_code' field of the JSON response, please ensure that line breaks and indentation are properly reflected.",
                        ]) 

                elif label == 'others': # 'macro_func', 'data_type', 'global_var' - these three

                    prompt.extend(["Now please cross-reference the C code before translation with the Rust code after translation, add a new key 'rust_code' to the metadata elements of each block in the JSON format of the original C code, and enter the translated code for each element.", #"The following Rust code has been translated from the original C source code and compiled successfully.",
                                    "The original C source code retains the following JSON-formatted metadata related to the C element named 'c_name', enclosed by the start line (c_start_line) and end line (c_end_line).",
                                    #"Now please cross-reference the C code before translation with the translated Rust code, add two new key to all of the elements (which are identified by element_id) respectively: 'rust_start_line' indicating where the Rust logical block including the corresponding Rust code begins, and 'rust_end_line' indicating where it ends.",
                                    "If there is no corresponding item, set the \"have_correspondence\" flag to False (and set rust_name, rust_start_line, and rust_end_line to None). If there is, set the \"have_correspondence\" flag to True."
                                    "If the have_correspondence flag is False, write a code comment in the no_equivalent_reason field explaining why an equivalent Rust implementation does not exist.",
                                    #"Cross-reference the C code before translation with the Rust code after translation, add a new key 'rust_code' to the metadata of each element block in the JSON format of the original C code, and enter the translated code for each element.",
                                    "When responding, please follow the response rules below and generate your response by selecting only one of the following three modes."

                    ])

                    prompt.extend(["", "## Response rules:",
                                "- Please determine the line numbers (rust_start_line and rust_end_line) based on logical Rust code blocks rather than direct line-by-line correspondence. A block should represent a complete logical unit such as an entire struct definition, conditional block and so on.",# This ensures the translated code maintains its semantic structure rather than being split mid-block.",
                                "- For variables and fields that are part of a struct, please provide the line numbers for the entire struct definition block that contains them, not just the individual line numbers.",
                                #"- If there is no corresponding item, set the \"have_correspondence\" flag to False (and set rust_name, rust_start_line, and rust_end_line to None). If there is, set the \"have_correspondence\" flag to True."
                                "- Please provide responses only for the Rust code equivalents of the JSON metadata shown below. Do not respond with code elements other than the JSON metadata shown below.",
                                f"- Please provide your answer by writing the JSON content of the {answer_path} as 'modified_data' in 'modify_data' mode.",
                                f"- Since {answer_path} will continue to be updated, set 'start_line' to 1 and 'end_line' to -1 for the {answer_path} modification area.",
                                "- When writing the translated code in the 'rust_code' field of the JSON response, please ensure that line breaks and indentation are properly reflected.",
                        ])  
                    
                elif label == 'function': # 'func_def
                    prompt.extend(["Now please cross-reference the C code before translation with the Rust code after translation, add a new key 'rust_code' to the metadata elements of each block in the JSON format of the original C code, and enter the translated code for each element.", #"The following Rust code has been translated from the original C source code and compiled successfully.",
                                "The original C source code retains the following JSON-formatted metadata related to the function named 'name', enclosed by the start line (start_line) and end line (end_line).",
                                #"Now please cross-reference the C code before translation with the Rust code after translation, and add a new key 'rust_code' to each function entry in the JSON metadata.",
                                "When responding, please follow the response rules below and generate your response by selecting only one of the following three modes.",
                                # "For the 'rust_start_line' key, write the line number in the Rust code where each 'rust_code' appears in the following Rust code after translation.",
                                #"Provide the response in the value of the 'answer' key in JSON format.",
                                ])

                    prompt.extend(["", "## Response rules:", 
                                "- Please provide responses only for the Rust code equivalents of the JSON metadata shown below. Do not respond with code elements other than the JSON metadata shown below.",
                                "- As the value for the 'rust_code' key, write only the function signature for each function or macro function, excluding the function body entirely, ensuring that input and output types are visible.",
                                "- In addition, as the value for the 'rust_function_name' key, write only the function name.",
                                "- If there is no corresponding item, set the \"have_correspondence\" flag to False. If there is, set the \"have_correspondence\" flag to True.",
                                "- If the have_correspondence flag is False, write a code comment in the no_equivalent_reason field explaining why an equivalent Rust implementation does not exist.",
                                f"- Please provide your answer by writing the JSON content of the {answer_path} as 'modified_data' in 'modify_data' mode.",
                                f"- Since {answer_path} will continue to be updated, set 'start_line' to 1 and 'end_line' to -1 for the {answer_path} modification area.",
                                "- When writing the translated code in the 'rust_code' field of the JSON response, please ensure that line breaks and indentation are properly reflected.",
                    ])
                """

                prompt.extend([f"- To avoid hitting the token limit, keep the JSON data included in one response within {output_max} tokens, and provide only the first part of the JSON data with clear separation for now, even if the response will be split into multiple parts.", # If it's a long response,
                    f"- If the response is split into multiple parts and there is still remaining JSON data, write a boolean value of True for the 'ongoing' key. If the JSON data is the final part, write a boolean value of False for the 'ongoing' key.",
                    ])
                
                prompt.extend(["- Please include the element name (\"name\"), start line (\"start_line\"), and end line (\"end_line\") of the C source code as is in the JSON format response.",
                               #"- When representing backslashes as byte literals, escape the backslash twice in the source code, and also escape it again in the byte literal, resulting in four backslashes (double backslashes).",
                               #"- When representing backslashes as character literals, escape the backslash once in the source code and again in the character literal, resulting in two backslashes.",
                               ])
                #prompt.extend(["The format of the response will be as follows:", corresp_format])
                
                prompt.extend(["\n## Response format", "In summary, please respond in the following JSON format:"]) 
                #prompt.extend([json_template])
                #prompt.extend([autonomous_template])
                prompt.extend([ask_map_template])

            
                """
                #tmp_rust_path = create_rust_base_json(c_path, meta_dir, label, div_start_line)
                tmp_rust_path = create_tmp_rust_json(c_path, meta_dir, label) # , div_start_line
                tmp_rust_path = transform_tmp(tmp_rust_path, label) # Added this
                tmp_json_data = read_file(tmp_rust_path) # Reading temporarily #Also, json data cannot be embedded into the prompt list when read as json, so using read_file().
                """

                #c_json_data = obtain_metadata(c_path, meta_dir, False, False)
                
                rust_code = get_lined_code(rust_path, work_dir)
                #c_code = get_lined_code(c_path, work_dir)  # rust_add_line_numbers("tmp_c.c", modified_start_line)
                c_code = get_unit_code_with_location(one_unit, database_dir) # , original_dir, target_dir

                # Removed this
                # if modified_files:
                #     prompt.extend(["", "## Rust files that were modified:"])
                # for mod_path in modified_files:
                #     prompt.append(f"    - {mod_path}")


                #prompt.extend(["## The converted Rust source code:", rust_code])
                prompt.extend(["", f"## The original C source code:", c_code])

                """
                if c_path is not None:
                    prompt.extend(["", f"## The original C source code:", c_code])
                else:
                    prompt.extend(["", f"## The original C source code ({c_path}):", c_code])  # inserted_c_code])
                """
                prompt.extend([f"## The converted Rust source code:", rust_code]) # ({rust_path})

                #prompt.extend(["", f"## The original C JSON-formatted metadata for {c_path} ({label} blocks):", tmp_json_data])
                prompt.extend(["", f"## The original C JSON-formatted metadata:", tmp_json_data])

            else:
                if error is None:
                    # prompt.extend(["Please continue providing the JSON data that maps between the C and Rust code.",
                    #     "The format of the response will be as follows:",
                    #     corresp_format])
                    prompt = []
                    prompt.extend(["Please continue providing the JSON data that maps between the C and Rust code.",
                                #"- When representing backslashes as byte literals, escape the backslash twice in the source code, and also escape it again in the byte literal, resulting in four backslashes (double backslashes).",
                               #"- When representing backslashes as character literals, escape the backslash once in the source code and again in the character literal, resulting in two backslashes.",
                              
                                    ])

                else:
                    if ongoing_flag is False:
                        prompt = []
                        prompt.extend(["Please continue providing the JSON data that maps between the C and Rust code.",
                                #"- When representing backslashes as byte literals, escape the backslash twice in the source code, and also escape it again in the byte literal, resulting in four backslashes (double backslashes).",
                               #"- When representing backslashes as character literals, escape the backslash once in the source code and again in the character literal, resulting in two backslashes.",
                              
                                        ])
                    else:
                        prompt = []
                        prompt.extend(["Please continue providing the JSON data that maps between the C and Rust code.",
                                #"- When representing backslashes as byte literals, escape the backslash twice in the source code, and also escape it again in the byte literal, resulting in four backslashes (double backslashes).",
                               #"- When representing backslashes as character literals, escape the backslash once in the source code and again in the character literal, resulting in two backslashes.",
                              
                                         ])

                prompt.extend(["Please select only one of the three modes when responding and generate your response accordingly."])
                prompt.extend(["\n## Response format", "Please respond in the following JSON format:"]) 
                prompt.extend([ask_map_template])

        
        if repair_target == "ask_unimplemented":

            if repair_count == 1:

                prompt = []
                prompt.extend(["Now please cross-reference the C code before translation with the Rust code after translation, add a new key 'rust_code' to the metadata elements of each block in the JSON format of the original C code, and enter the translated code for each element.", #"The following Rust code has been translated from the original C source code and compiled successfully.",
                                "Please identify elements in the converted Rust code that are marked as 'unimplemented'",
                                f"For the unimplemented elements you find, write your answer in the specified JSON format to {answer_path}.",
                                #"Cross-reference the C code before translation with the Rust code after translation, add a new key 'rust_code' to the metadata elements of each block in the JSON format of the original C code, and enter the translated code for each element.",
                                "When responding, please follow the response rules below and generate your response by selecting only one of the following three modes.",
                                ])

                prompt.extend(["", "## Response rules:",
                            "- For each unimplemented element in the Rust code, please include the file path, start line, and end line as values for their respective keys (\"rust_file_path\", \"rust_start_line\", \"rust_end_line\").",
                            "- Please determine the line numbers (rust_start_line and rust_end_line) based on logical Rust code blocks rather than direct line-by-line correspondence. A block should represent a complete logical unit such as an entire struct definition, conditional block and so on.",# This ensures the translated code maintains its semantic structure rather than being split mid-block.",
                            #"- Please provide responses only for the Rust code equivalents of the JSON metadata shown below. Do not respond with code elements other than the JSON metadata shown below.",
                            "- If there is a corresponding item in the original C code, set the \"have_correspondence\" flag to True, and set c_name and c_file_path as the element (function and so on) name in the original C code and its file path. If there is not, set the \"have_correspondence\" flag to False."
                            "- If the have_correspondence flag is False, write a code comment in the no_equivalent_reason field explaining why an equivalent Rust implementation does not exist.",
                            f"- Please provide your answer by writing the JSON content of the {answer_path} as 'modified_data' in 'modify_data' mode.",
                            f"- Since {answer_path} will continue to be updated, set 'start_line' to 1 and 'end_line' to -1 for the {answer_path} modification area.",
                            "- When writing the translated code in the 'rust_code' field of the JSON response, please ensure that line breaks and indentation are properly reflected.",
                    ])
                
                prompt.extend(["\n## Response format", "In summary, please respond in the following JSON format:"]) 
                #prompt.extend([json_template])
                prompt.extend([autonomous_template])

                prompt.extend([f"- Please use the following JSON format when writing to {answer_path}:"]) 
                prompt.extend([unimpl_template])


                rust_code = get_lined_code(rust_path, work_dir)
                c_code = get_lined_code(c_path, work_dir)  # rust_add_line_numbers("tmp_c.c", modified_start_line)

                prompt.extend(["", f"## The original C source code ({c_path}):", c_code])  # inserted_c_code])
                prompt.extend([f"## The converted Rust source code:", rust_code]) #  ({rust_path})

                #prompt.extend(["", f"## The JSON-formatted metadata for {c_path} ({label} blocks):", tmp_json_data])

            
            else:
                if error is None:
                    prompt = []
                    prompt.extend(["Please continue providing the JSON data that identifies unimplmented Rust code.",])

                else:
                    if ongoing_flag is False:
                        prompt = []
                        prompt.extend(["Please continue providing the JSON data that identifies unimplmented Rust code.",])

                    else:
                        prompt = []
                        prompt.extend(["Please continue providing the JSON data that identifies unimplmented Rust code."])

                prompt.extend(["Please select only one of the three modes when responding and generate your response accordingly."])



        if repair_target == "judge_conds":
            conds_status_path = interface.conds_status_path
            conds_data = read_json(conds_status_path)

            macro_list = []
            for macro, entry in conds_data.items():
                for def_item in entry['timeline']:
                    macro_list.append({
                        "name": macro,
                        "value": def_item['value']
                    })

            build_command = "cargo build --release"

            if repair_count == 1:
                prompt = ["Currently, macro variables used for conditional compilation in C code are defined and used as shown below, and we intend to convert these macro variables to Rust.",
                        "For each macro variable, please determine which location (macro type) is appropriate as the definition location when converted to Rust, and provide your answer.",
                        "Please follow the response rules and response types below when answering."]

                prompt.extend(["## Response rules:", 
                            f"- Please provide your answer by writing to {answer_path} using modify_data mode.",
                            "- Please select one macro type from the following options.", 
                            f"- Since the build command is executed without options as {build_command}, do not use build command options (--features, --cfg, etc.) for enabling. Instead, enable them within Cargo.toml, build.rs, lib.rs, and their respective module files.",
                            "- If you need more information beyond the macro variable details below, please refer to the source code using read_data mode.",
                            "- If errors occur when representing backslashes as byte literals, you need to escape the backslash in the source code and also escape it again in the byte literal, so use three backslashes (double backslashes).",
                            "- If errors occur when representing backslashes as character literals, you need to escape the backslash in the source code and also escape it again in the character literal, so use two backslashes.",
                            f"- To avoid hitting the output token limit, keep the JSON data included in a single response within {output_max} tokens.", 
                            "- If the JSON data in a single mode response is likely to exceed the token limit, split it across multiple responses.",
                            "- If the JSON data is the final part, set the 'ongoing_in_mode' key to a boolean value of False. If there is remaining JSON data, set the 'ongoing_in_mode' key to a boolean value of True.",
                            "- Always create responses in a single mode (`read_data`, `modify_data`, `execute_command`), and use `ongoing_in_mode` only when further interaction is needed within that single mode.",
                            "- When switching modes, end the response by requesting a new request.",
                            "- The content of modified_code answered in `modify_data` mode will be directly copy-pasted, so it must absolutely not contain any omitted sections.",
                            ])
                prompt.extend([f"- The content to be written to {answer_path} should be in the following JSON format."])
                prompt.extend([judge_template])

                prompt.extend(["## Macro Types:",
                                "1. CARGO",
                                "- Defined in Cargo.toml.",
                                "- Defines feature flags and crate-level configurations.",
                                "- Defines values that serve as criteria for compile-time conditional branching.",
                                "",
                                "2. BUILD",
                                "- Defined in build.rs.",
                                "- Defines build system level configurations necessary for compile-time environment detection, system-dependent flag settings, and external library linking settings.",
                                "",
                                "3. LIB",
                                "- Defined in lib.rs.",
                                "- Defines constants and macros used throughout the crate.",
                                "- Defines configuration values shared between modules.",
                                "",
                                "4. MODULE",
                                "- Defined within the Rust module corresponding to the original C file.",
                                "- Definitions within the Rust module corresponding to the original C file, such as module-specific conditional branching and settings limited to specific implementations."
                            ])

                if macro_list:
                    sentence = []
                    prompt.extend(["", "## Macro variables used for conditional compilation in C:"])
                    for item in macro_list:
                        sentence.append(f"{item['name']}: {item['value']}")
                    prompt.extend(sentence)

                prompt.extend(["", "## Target C program:"])
                directory_structure = get_dir_struct("translation", raw_dir, None)  #rust_output_dir)
                prompt.extend([directory_structure, ""])


            else:
                if ongoing_flag is False:
                    prompt = []
                    prompt.extend(["Please continue providing the answer regarding the definition location (macro type) when converting C conditional compilation to Rust."])
                else:
                    prompt = []
                    prompt.extend(["Please continue providing the answer on the definition location (macro type) when converting C conditional compilation to Rust."])


        if repair_target == "judge_macros":  # Used commonly across multiple files
            conds_status_path = interface.conds_status_path
            conds_data = read_json(conds_status_path)

            macro_list = []
            for macro, entry in conds_data.items():
                for def_item in entry['timeline']:
                    macro_list.append({
                        "name": macro,
                        "value": def_item['value']
                    })

            build_command = "cargo build --release"
            if repair_count == 1:
                prompt = ["Currently, macro variables commonly used across multiple files in C code are defined and used as shown below, and we intend to convert these macro variables to Rust.",
                        "For each macro variable, please determine which location (macro type) is appropriate as the definition location when converted to Rust, and provide your answer.",
                        "Please follow the response rules and response types below when answering."]

                prompt.extend(["## Response rules:", 
                            "- Please select one macro type from the following options.", 
                            f"- Since the build command is executed without options as {build_command}, do not use build command options (--features, --cfg, etc.) for enabling. Instead, enable them within Cargo.toml, build.rs, lib.rs, and their respective module files.",
                            f"- Please provide your answer by writing to {answer_path} using modify_data mode.",
                            "- If you need more information beyond the macro variable details below, please refer to the source code using read_data mode.",
                            "- If errors occur when representing backslashes as byte literals, you need to escape the backslash in the source code and also escape it again in the byte literal, so use three backslashes (double backslashes).",
                                "- If errors occur when representing backslashes as character literals, you need to escape the backslash in the source code and also escape it again in the character literal, so use two backslashes.",
                                f"- To avoid hitting the output token limit, keep the JSON data included in a single response within {output_max} tokens.",
                                "- If the JSON data in a single mode response is likely to exceed the token limit, split it across multiple responses.",
                                "- If the JSON data is the final part, set the 'ongoing_in_mode' key to a boolean value of False. If there is remaining JSON data, set the 'ongoing_in_mode' key to a boolean value of True.",
                                "- Always create responses in a single mode (`read_data`, `modify_data`, `execute_command`), and use `ongoing_in_mode` only when further interaction is needed within that single mode.",
                                "- When switching modes, end the response by requesting a new request.",
                                "- The content of modified_code answered in `modify_data` mode will be directly copy-pasted, so it must absolutely not contain any omitted sections.",
                                ])
                
                prompt.extend([f"- The content to be written to {answer_path} should be in the following JSON format."])
                prompt.extend([judge_template])

                prompt.extend(["## Macro Types:",
                                "1. CARGO",
                                "- Defined in Cargo.toml.",
                                "- Defines feature flags and crate-level configurations.",
                                "- Defines values that serve as criteria for compile-time conditional branching.",
                                "",
                                "2. BUILD",
                                "- Defined in build.rs.",
                                "- Defines build system level configurations necessary for compile-time environment detection, system-dependent flag settings, and external library linking settings.",
                                "",
                                "3. LIB",
                                "- Defined in lib.rs.",
                                "- Defines constants and macros used throughout the crate.",
                                "- Defines configuration values shared between modules.",
                                "",
                                "4. MODULE",
                                "- Defined within the Rust module corresponding to the original C file.",
                                "- Definitions within the Rust module corresponding to the original C file, such as module-specific conditional branching and settings limited to specific implementations."
                            ])

                if macro_list:
                    sentence = []
                    prompt.extend(["", "## Macro variables in C:"])
                    for item in macro_list:
                        sentence.append(f"{item['name']}: {item['value']}")
                    prompt.extend(sentence)

                prompt.extend(["", "## Target C program:"])
                directory_structure = get_dir_struct("translation", raw_dir, None)  #rust_output_dir)
                prompt.extend([directory_structure, ""])
        
            else:
                if ongoing_flag is False:
                    prompt = []
                    prompt.extend(["Please continue providing the answer regarding the definition location (macro type) when converting C conditional compilation to Rust."])
                else:
                    prompt = []
                    prompt.extend(["Please continue providing the answer on the definition location (macro type) when converting C conditional compilation to Rust."])



        if execute_error is not None or execute_out is not None:
            if execute_out is not None:
                prompt.extend(["\n## Execution result:",
                              "The result executed in execute_command mode is as follows:",
                            f"{execute_out}",
                            ""])

            if execute_error is not None:
                prompt.extend(["## Execution result error: ",
                               "The error result executed in execute_command mode is as follows:",
                            f"{execute_error}"])
            execute_error = None # Initialization
            execute_out = None
        
        if read_prompt is not None:
            prompt.extend(["", "## Response to the previous request:"]) 
            prompt.extend(read_prompt)
            read_prompt = None # Initialization

        #prompt = get_auto_prompt(prompt, execute_path)
        print(f"ongoing_flag is {ongoing_flag}")

        if ongoing_flag is False and repair_target != "ask_correspondence" and repair_target != "ask_generates": # and repair_target != "judge_conds" and repair_target != "judge_macros":
            
            prompt.extend(["",
                                "## Response Modes:",
                                "1. In 'read_data' mode:",
                                "### Purpose:",
                                "- Returns the content of the specified file as it is.",
                                "### Response Format:",
                                "- Include the file path you want to read inside the \"target_files\" key in the JSON data.",
                                "- If the number of lines in the file you want to read is too large and cannot be viewed due to context window limitations, you can specify \"start_line\" and \"end_line\" along with file_path in \"file_slices\" to know about that specific range of lines.",
                                "",
                                "2. In 'modify_data' mode:",
                                "### Purpose:",
                                "- Allows modifying an existing file at the specified file path.",
                                "### Response Format:",
                                "- To accurately identify the parts to be modified, make sure to always read the target file in read_data mode before executing modify_data mode.",
                                "- Please insert the filename, start line, and end line of the section to be deleted into the \"file_path\", \"start_line\", and \"end_line\" keys in the JSON data.",                                              
                                "- Then, insert the new content that should be inserted at that [start_line] into the value of the \"modified_data\" key.",
                                "- Detailed modification process is as follows. Please carefully write start_line, end_line and modified_data considering the process:",
                                "    1. All code in the specified range (from [start_line] to [end_line]) will be completely deleted.",
                                "    2. The content you provide in \"modified_data\" will be inserted at [start_line].",
                                "    3. All code from [end_line + 1] onwards will remain unchanged and be appended after your modified_data.",
                                "- Please use the exact line numbers and indentation levels shown on the left side of the code (Line X [Y], where X is the line number and Y is the indentation level) for start_line, end_line and modified_data.",
                                "- In case the modification content (modified_data) for a single range (start_line-end_line) is too long to include in one entry:",
                                "    - Please split it across multiple answer entries.",
                                "    - Each of these answer entries should maintain the same file_path, start_line, and end_line values",
                                "    - Include modification_part representing the number of the current part in each entry to track the split.",
                                #"    - Include modification_part information in each entry to track the split:",
                                #"        - current: the number of the current part:",
                                #"        - total: the total number of parts",
                                "    - please remain ongoing_in_mode and ongoing flags true until all parts are delivered",
                                #"- Do not propose multiple modifications for the same line.",
                                #"- Do not split your modifications across multiple entries in the answer array when they target the exact same line range. For each unique (start_line, end_line) pair, there should be only ONE modification entry in your answer.",
                                "- Set the ready_to_execute flag to True if this response marks the end of a coherent modification set and it's ready to be tested",
                                #"- The 'Line X:' prefix is not part of the actual code - they're just line indicators. When providing modified_data, please use the correct indentation from the original code, ignoring the prefix.",
                                #"- Insert the filename, start line, and end line of the change location to be inserted into the values of the \"file_path\", \"start_line\", and \"end_line\" keys in JSON format data.",
                                f"- For file_path, write a relative path in the format '{work_dir}/path/to/file'", #f"- For file_path, write a relative path starting from {work_dir}.",
                                "- Insert appropriate indentation in modified_data so that it can be executed correctly when copied and pasted into the original code's location from start_line to end_line.",
                                "- \"modified_data\" must not contain any omissions and must strictly maintain the appropriate indentation, as it will be directly inserted and executed in the original code.",
                                "- Ensure that \"modified_data\" follows the exact indentation level [Y] shown for each line in the original code.",
                                "- If you want to only perform deletion without inserting into a specific location in the existing specified file path, set the value of 'is_deletion' to True.",
                                #"- If you want to overwrite the entire file rather than just modifying the specified line range, set the value of 'overwrite_all' to True.",
                                "- Set the value of \"no_simplification\" to True if the functionality intended before modification exists completely without any omission and simplification. Set it to False otherwise.",
                                "- If the target file for editing is a JSON file, set the \"is_JSON\" flag to True and insert the modified JSON data into \"modified_data\"",
                                #""
                                #"- Since I will ask about the actual modifications later, for now, please only specify the \"start_line\" and \"end_line\" that need modification. Do not write \"modified_data\".",
                                "",
                                "3. In 'execute_command' mode:",
                                "### Purpose:",
                                "- Executes the provided shell script code.",
                                "### Response Format:",
                                f"- This executes separately from {run_path}. If not necessary beyond {run_path}, you do not need to include it in the response.",
                                "- Put the shell script code to be executed in the \"answer\" field of the JSON format data.",
                                f"- The answered shell script code will be saved in the shell script file at {execute_path} and executed in the {execute_dir} directory.",
                                "- The execution of ./execute.sh should not have any arguments.",
                                "- The shell script can include multiple commands."
                            ])
            
            if repair_count != 1 and repair_target == "build":
                prompt.extend([#"",
                        "",
                        "",
                        "## Other notes:", 
                        "- The Rust program is generated as a library crate, so please use only src/lib.rs and do not create src/main.rs.",
                        "- Since other elements will be converted step by step afterwards, please convert only the specified code elements and do not add extra function definitions.",
                        "- For the same error, do not suggest a solution that has already been tried once. Please suggest a different solution.",
                        f"- To avoid hitting the output token limit, keep the JSON data included in a single response within {output_max} tokens.",  
                        "- If the JSON data in a single mode response is likely to exceed the token limit, split it across multiple responses.",
                        "- If the JSON data is the final part, set the 'ongoing_in_mode' key to a boolean value of False. If there is remaining JSON data, set the 'ongoing_in_mode' key to a boolean value of True.",
                        "- Always create responses in a single mode (`read_data`, `modify_data`, `execute_command`), and use `ongoing_in_mode` only when further interaction is needed within that single mode.",
                        "- When switching modes, end the response by requesting a new request.",
                        "- The content of modified_data answered in `modify_data` mode will be directly copy-pasted and executed, so it must absolutely not contain any omitted sections.",
                        ])
                    
            prompt.extend(["\n## Response format", "In summary, please respond in the following JSON format:"]) 
            prompt.extend([autonomous_template])
            
            #prompt.extend([see_template])
        

            
            if error is not None and error is not True:
                #if repair_count != 1 and repair_target == "build":
                #    prompt.extend([f"## Error that occurred when executing the Rust program (executing the shell script at {run_path}):", error])
                
                prompt.extend(["", "## Error:", error])

            if repair_target == "compile":
                prompt.extend(["", "## Directory structure of the target Rust program:"])
                directory_structure = get_dir_struct("translation", work_dir, None)  #rust_output_dir)
                prompt.extend([directory_structure, ""])

                if W_O_DEP:
                    prompt.extend(["", "## Module structure of the target Rust Program:"])
                    structure = get_cargo_modules(rust_output_dir)
                    prompt.extend([structure, ""])


            if repair_target == "ask_generates":
                prompt.extend(["", "## Directory structure of the translated Rust program:"])
                directory_structure = get_dir_struct("translation", work_dir, None)  #rust_output_dir)
                prompt.extend([directory_structure, ""])
                
                prompt.extend(["", "## Directory structure of the original C program:"])
                directory_structure = get_dir_struct("translation", raw_dir, None)
                prompt.extend([directory_structure, ""])
                

            
        ################################################

        prompt = adjust_prompt(prompt)
        print("-------------------------")

        print(f"repair_target: {repair_target}")

        if repair_target == "ask_generates" or repair_target == "ask_correspondence": # Memory must not be interrupted when asking for correspondence       
            delete_file(execute_path)
            create_file(execute_path)
            if ask_count == 1:
                rsp_json = ask_llm(prompt, "init", llm_interface)
            else:
                rsp_json = ask_llm(prompt, "continue", llm_interface)
            
            ask_count += 1
            
        else:
            if repair_count == 1:
                rsp_json = ask_llm(prompt, "init", llm_interface)
            else:
                delete_file(execute_path)
                create_file(execute_path)
                rsp_json = ask_llm(prompt, "continue", llm_interface)


        ################################################
        #ongoing_flag = False
        ongoing_in_mode_flag = False

        sum_target_list = []
        sum_modified_list = []
        sum_deleted_list = []

        sum_slice_list = []

        while (1):
            prompt = []
            execute_error = None
            if 'mode' in rsp_json:
                mode = rsp_json['mode']

                if mode == 'read_data':
                    if 'answer' in rsp_json:
                        code = rsp_json['answer']
                        append_file(execute_path, code)

                    if 'target_files' in rsp_json:
                        target_list = rsp_json['target_files']
                        if not isinstance(target_list, list):
                            target_list = [target_list]
                        sum_target_list.extend(target_list)

                    if 'file_slices' in rsp_json and rsp_json['file_slices'] is not None:
                        slice_list = rsp_json['file_slices']
                        if not isinstance(slice_list, list):
                            slice_list = [slice_list]
                        sum_slice_list.extend(slice_list)


                if mode == 'modify_data':
                    if 'answer' in rsp_json:
                        modified_list = rsp_json['answer'] # Could also put individually converted items here
                        if not isinstance(modified_list, list):
                            modified_list = [modified_list]
                        #sum_modified_list.extend(modified_list)

                        if repair_target != "compile":
                            sum_modified_list.extend(modified_list)

                        else:
                            sequences = []
                            seen_sequences = set()
                            for mod in modified_list:
                                if mod['file_path'] not in seen_sequences:
                                    sequences.append(mod['file_path'])
                                    seen_sequences.add(mod['file_path'])

                            for seq_path in sequences:
                                prompt = []                                
                                prompt.extend([f"Please provide the actual modified_data for the previously identified locations in {seq_path} file using modify_data mode.", #f"Please write the actual modifications for the {seq_path} file in modify_data mode.",
                                                "",
                                                "## Response rules:", 
                                                "- Please insert the filename, start line, and end line of the section to be deleted into the \"file_path\", \"start_line\", and \"end_line\" keys in the JSON data.",
                                                "- Then, insert the new content that should be inserted at that [start_line] into the value of the \"modified_data\" key.",
                                                "- Detailed modification process is as follows. Please carefully write start_line, end_line and modified_data considering the process:",
                                                "    1. All code in the specified range (from [start_line] to [end_line]) will be completely deleted.",
                                                "    2. The content you provide in \"modified_data\" will be inserted at [start_line].",
                                                "    3. All code from [end_line + 1] onwards will remain unchanged and be appended after your modified_data.",
                                                #"- Please use the exact line numbers shown on the left side of the code (Line X) for start_line and end_line.",
                                                "- Please use the exact line numbers and indentation levels shown on the left side of the code (Line X [Y], where X is the line number and Y is the indentation level) for start_line, end_line and modified_data.",
                                                "- In case the modification content (modified_data) for a single range (start_line-end_line) is too long to include in one entry:",
                                                "    - Please split it across multiple answer entries.",
                                                "    - Each of these answer entries should maintain the same file_path, start_line, and end_line values",
                                                "    - Include modification_part representing the number of the current part in each entry to track the split.",
                                                #"    - Include modification_part information in each entry to track the split:",
                                                #"        - current: the number of the current part:",
                                                #"        - total: the total number of parts",
                                                "    - please remain ongoing_in_mode and ongoing flags true until all parts are delivered",
                                                #"- Do not propose multiple modifications for the same line.",
                                                #"- Do not split your modifications across multiple entries in the answer array when they target the exact same line range. For each unique (start_line, end_line) pair, there should be only ONE modification entry in your answer.",
                                                "- Set the ready_to_execute flag to True if this response marks the end of a coherent modification set and it's ready to be tested",
                                                #"- The 'Line X:' prefix is not part of the actual code - they're just line indicators. When providing modified_data, please use the correct indentation from the original code, ignoring the prefix.",
                                                f"- For file_path, write the relative path starting from {work_dir}.",
                                                "- Insert appropriate indentation in modified_data so that it can be executed correctly when copied and pasted into the original code's location from start_line to end_line.",
                                                "- \"modified_data\" must not contain any omissions and must strictly maintain the appropriate indentation, as it will be directly inserted and executed in the original code.",
                                                "- Ensure that \"modified_data\" follows the exact indentation level [Y] shown for each line in the original code.",
                                                "- If you want to only perform deletion without inserting into a specific location in the existing specified file path, set the value of 'is_deletion' to True.",
                                                #"- If you want to overwrite the entire file rather than just modifying the specified line range, set the value of 'overwrite_all' to True.",
                                                "- Set the value of \"no_simplification\" to True if the functionality intended before modification exists completely without any omission and simplification. Set it to False otherwise.",
                                                "- If the target file for editing is a JSON file, set the \"is_JSON\" flag to True and insert the modified JSON data into \"modified_data\"",
                                                #""
                                                "- In modifications, do NOT use unsafe or raw pointers. Instead, please use appropriate safe alternatives like Box, Arc, Vec, and others. Depending on your need, use the Rust standard library or crates to achieve equivalent functionality in a safe manner.",
                                                "- When representing backslashes as byte literals, escape the backslash twice in the source code, and also escape it again in the byte literal, resulting in four backslashes (double backslashes).",
                                                "- When representing backslashes as character literals, escape the backslash once in the source code and again in the character literal, resulting in two backslashes.",
                                                "- After translating functions to Rust, please add the prefix \"rust_\" to all function names.",
                                ])

                                prompt.extend(["\n## Response format", "In summary, please respond in the following JSON format:"])
                                prompt.extend([modify_template])

                                seq_code = get_lined_code(seq_path, work_dir)  #read_file(seq_path)
                                prompt.extend([f"## Code in {seq_path}:", seq_code])

                                child_rsp_json = ask_llm(prompt, "continue", llm_interface)

                                if 'answer' in child_rsp_json:
                                    modified_list = child_rsp_json['answer'] # Could also put individually converted items here
                                    if not isinstance(modified_list, list):
                                        modified_list = [modified_list]
                                    sum_modified_list.extend(modified_list)
                            
                            prompt = []

                
                if mode == 'delete_data':
                    if 'answer' in rsp_json:
                        deleted_list = rsp_json['answer'] # Could also put individually converted items here
                        if not isinstance(deleted_list, list):
                            deleted_list = [deleted_list]
                        sum_deleted_list.extend(deleted_list)

                if mode == 'execute_command':
                    if 'answer' in rsp_json:
                        code = rsp_json['answer']
                        append_file(execute_path, code)
        

            print(f"ongoing_flag is {ongoing_flag}")
            if 'ongoing' in rsp_json:
                ongoing_flag = rsp_json['ongoing']

            if 'ongoing_in_mode' in rsp_json:
                ongoing_in_mode_flag = rsp_json['ongoing_in_mode']

            if ongoing_in_mode_flag is False:
                break

            print("Keep going to receive Rust code in modifying.")
            
            if repair_target == "build":
                if repair_count == 1:
                    prompt = [f"Please continue providing the JSON data response with the converted Rust code."]
                else:
                    prompt = [f"Please continue providing the JSON data response for resolving the compilation error."]

            else: # Just roughly summarized here.
                prompt = [f"Please analyze the C code elements and provide the corresponding Rust code information for the remaining JSON items listed below."] #Please provide responses corresponding to the JSON data listed below."]

            # This might not be needed
            prompt.extend(["", "## Response rules:", 
                        f"- To avoid exceeding the output token limit, ensure that the JSON data included in a single response does not exceed {output_max} tokens.",
                        "- The modified code must not contain any omission and be complete to ready to copy-paste for execution. If the JSON data in one response is likely to exceed the token limit, split the response into multiple parts.",
                        "- If this is the final part of the JSON data, set the 'ongoing' key to a boolean value of False. If there is remaining JSON data, set the 'ongoing' key to a boolean value of True.",            
                        "- Please respond in the same JSON format described before."
                        ])
            
            prompt.extend(["", "## Response Format:"])
            prompt.extend([autonomous_template])

            #tmp_data = read_json(tmp_rust_path)
            if isinstance(key_json, str):
                tmp_data = json.loads(key_json)
            else:
                tmp_data = key_json

            remaining_list = get_remaining_list(tmp_data, sum_modified_list, database_dir)

            if len(remaining_list) == 0:
                break

            remaining_path = f"{database_dir}/remaining.json"
            write_json(remaining_path, remaining_list)
            remaining_data = read_file(remaining_path)

            prompt.extend(["", "## Remaining items:"])
            prompt.extend([remaining_data])

            rsp_json = ask_llm(prompt, "continue", llm_interface) #code_blocks = extract_code_blocks(response)

        
        # tmp_data = read_json(tmp_rust_path)
        # remaining_list = get_remaining_list(tmp_data, sum_modified_list)

        # remaining_path = "remaining.json"
        # write_json(remaining_path, remaining_list)
        # remaining_data = read_file(remaining_path)
    

        ######################## Proceed file by file ########################

        print(f"Running program for the mode: {mode}")
        if mode == 'modify_data':
            print(f"In mode: {mode}")
            # Not including modifications for the same start_line, end_line. For example, in the case of split responses with start_line = 1, end_line = 600, start_line and end_line remain the same throughout.
            for item in sum_modified_list:
                if item['file_path'] != answer_path:
                    item['file_path'] = answer_path # added

            part_editied_files = reflect_line_modification(sum_modified_list, work_dir)
            #part_editied_files = reflect_line_modification(sum_modified_list, rust_output_dir) # execute_error =  #sum_modified_list.extend(added_list) #if MOD_LINE:
            editied_files.extend(part_editied_files)

            #if not reflect_success:
            #    return repair_count, error
        
        elif mode == 'read_data':
            print(f"In mode: {mode}")
            #output = run_read_script(execute_path, 50, True, None, "both")
            read_prompt = ["The content obtained in read_data mode is as follows.", ""] # Even if there was a previous response with the 'ongoing' flag set to true, this response must include None in the "answer" key of the JSON data as shown below.
                      # If the 'ongoing' flag is true, continue the response after returning None once.]
            #prompt.extend([none_format])

                        #"Command execution result: ",
                        #f"{output}", ""]
            
            for see_path in sum_target_list:
                file_code = get_lined_code(see_path, work_dir)
                read_prompt.extend([f"- Content of the file {see_path}:"])
                if len(file_code) == 0:
                    file_code = f"Line 1 [0]: [This {see_path} file is currently empty and contains no content. *** STOP *** Do not use read_data mode anymore.]"
                read_prompt.extend([f'{file_code}\n'])

            for see_item in sum_slice_list:
                file_code = get_lined_specific_code(work_dir, database_dir, see_item['file_path'], see_item['start_line'], see_item['end_line'])
                if len(file_code) == 0:
                    file_code = f"Line 1 [0]: [This {see_item['file_path']} file is currently empty and contains no content. *** STOP *** Do not use read_data mode anymore.]"
                read_prompt.extend([f"- Content of {see_item['start_line']} - {see_item['end_line']} lines in the file {see_item['file_path']}:"])
                if len(file_code) == 0:
                    file_code = f"Line 1 [0]: [This {see_item['file_path']} file is currently empty and contains no content. *** STOP *** Do not use read_data mode anymore.]"
                read_prompt.extend([f'{file_code}\n'])


            #rsp_json = ask_llm(prompt, "continue", interface)

            #print(rsp_json)
            print("End of rsp_json")
        
        elif mode == 'execute_command':
            print(f"In mode: {mode}")
            execute_error, execute_out, repair_count = run_script(execute_path, 50, True, None, "both", None, repair_count, None, None, mode)
            
        repair_count += 1
        #modified_file_list.extend(sum_modified_list)

    # Putting this on hold for now
    #check_dif(target_dir)

    iteration_dict[rust_path] = repair_count

    if repair_target == "build" or repair_target == "compile":
        seen_files = set()
        for edite_file in editied_files:
            if edite_file not in seen_files:
                seen_files.add(edite_file)
        #return editied_files
        return list(seen_files)
    
    elif repair_target == "ask_generates" or repair_target == "ask_correspondence":
        data = read_json(answer_path)
        write_json(answer_path, data)


        
def get_claude_model(llm_choice):

    claude_model = None
    if llm_choice == "claude_azure":
        # claude_model = 'databricks-claude-sonnet-4'
        #claude_model = 'databricks-claude-sonnet-4-5'
        #claude_model = 'databricks-claude-opus-4-5'
        claude_model = 'databricks-claude-opus-4-6'

    elif llm_choice == "claude":
        #claude_model = 'claude-sonnet-4-5-20250929'
        claude_model = 'claude-opus-4-6'

    return claude_model


