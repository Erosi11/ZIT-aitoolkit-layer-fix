#!/usr/bin/env python3
"""
LoRA Converter for z-image-turbo / Lumina2 (GUI Version)
Converts separated to_q/to_k/to_v LoRA layers into a merged qkv format.
Includes Batch Processing, Custom Save Directories, and Duplicate Skipping.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from collections import defaultdict
import traceback
import threading

try:
    import torch
    from safetensors.torch import load_file, save_file
    from safetensors import safe_open
except ImportError as e:
    messagebox.showerror("Dependencies Missing", f"Missing required libraries: {e}\n\nPlease run in terminal:\npip install torch safetensors")
    sys.exit(1)


def convert_lora(input_path, output_path, progress_callback=None):
    """Core logic to convert a single LoRA file."""
    if progress_callback:
        progress_callback(0, f"Loading: {os.path.basename(input_path)}")

    if input_path.endswith('.safetensors'):
        lora_dict = load_file(input_path)
    elif input_path.endswith(('.pt', '.pth')):
        lora_dict = torch.load(input_path, map_location='cpu')
    else:
        raise ValueError("Unsupported file format")

    total_keys = len(lora_dict)
    layer_groups = defaultdict(lambda: defaultdict(dict))
    output_dict = {}
    converted_count = 0

    processed = 0
    for key, value in lora_dict.items():
        processed += 1
        if progress_callback and processed % 100 == 0:
            progress_callback(10 + 40 * processed // total_keys, "Parsing keys...")

        # Renaming and merging logic
        if '.attention.to_out.0.' in key:
            new_key = key.replace('.to_out.0.', '.out.')
            output_dict[new_key] = value
            if 'lora_A' in key:
                base = key.rsplit('.lora_A', 1)[0]
                alpha_key = f"{base}.alpha"
                if alpha_key in lora_dict:
                    new_alpha = alpha_key.replace('.to_out.0.', '.out.')
                    output_dict[new_alpha] = lora_dict[alpha_key]
            continue

        if '.attention.to_' in key and '.alpha' in key:
            continue

        if '.attention.to_' in key and any(x in key for x in ('.to_q.', '.to_k.', '.to_v.')):
            parts = key.split('.')
            attn_type = next((p[3:] for p in parts if p in ('to_q', 'to_k', 'to_v')), None)
            lora_type = next((p for p in parts if p in ('lora_A', 'lora_B')), None)

            if attn_type and lora_type:
                base_parts = [p for p in parts if p not in ('to_q', 'to_k', 'to_v')]
                base_key = '.'.join(base_parts[:-2])
                layer_groups[base_key][attn_type][lora_type] = value
                continue

        output_dict[key] = value

    for base_key, qkv_dict in layer_groups.items():
        if all(x in qkv_dict for x in ('q', 'k', 'v')):
            try:
                qB, kB, vB = qkv_dict['q']['lora_B'], qkv_dict['k']['lora_B'], qkv_dict['v']['lora_B']
                qA, kA, vA = qkv_dict['q']['lora_A'], qkv_dict['k']['lora_A'], qkv_dict['v']['lora_A']
                
                h, r = qB.shape
                qkv_B = torch.zeros(3*h, 3*r, dtype=qB.dtype)
                qkv_B[:h, :r], qkv_B[h:2*h, r:2*r], qkv_B[2*h:, 2*r:] = qB, kB, vB
                qkv_A = torch.cat([qA, kA, vA], dim=0)

                output_dict[f"{base_key}.qkv.lora_B.weight"] = qkv_B
                output_dict[f"{base_key}.qkv.lora_A.weight"] = qkv_A
                converted_count += 1

                alpha_q = lora_dict.get(f"{base_key}.to_q.alpha") or lora_dict.get(f"{base_key}.to_q.lora_A.alpha")
                if alpha_q is not None:
                    output_dict[f"{base_key}.qkv.alpha"] = alpha_q * 3.0
            except: continue

    metadata = {'converted_for': 'z-image-turbo / Lumina2', 'script': 'batch_converter.py'}
    save_file(output_dict, output_path, metadata=metadata)
    return converted_count


class LoRAConverterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LoRA Batch Converter Pro")
        self.geometry("750x550")
        
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.single_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.single_tab, text="Single File")
        self.setup_single_tab()

        self.batch_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.batch_tab, text="Batch Processing")
        self.setup_batch_tab()

        log_frame = ttk.LabelFrame(self, text="Status & Logs")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.log_text = tk.Text(log_frame, height=10, state='disabled', font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        self.progress = ttk.Progressbar(self, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)

    def log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.update_idletasks()

    def setup_single_tab(self):
        f = ttk.Frame(self.single_tab, padding=20)
        f.pack(fill=tk.BOTH)
        self.in_file = tk.StringVar()
        self.out_file = tk.StringVar()
        
        ttk.Label(f, text="Input LoRA:").grid(row=0, column=0, sticky='w')
        ttk.Entry(f, textvariable=self.in_file, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(f, text="Browse", command=self.browse_single_in).grid(row=0, column=2)
        
        ttk.Label(f, text="Output Path:").grid(row=1, column=0, sticky='w', pady=10)
        ttk.Entry(f, textvariable=self.out_file, width=60).grid(row=1, column=1, padx=5)
        ttk.Button(f, text="Browse", command=self.browse_single_out).grid(row=1, column=2)
        
        ttk.Button(f, text="Convert Single File", command=self.run_single).grid(row=2, column=1, pady=20)

    def setup_batch_tab(self):
        f = ttk.Frame(self.batch_tab, padding=20)
        f.pack(fill=tk.BOTH)
        
        self.batch_in_dir = tk.StringVar()
        self.batch_out_dir = tk.StringVar()
        self.skip_existing = tk.BooleanVar(value=True)
        
        # Input Directory
        ttk.Label(f, text="Source Folder:").grid(row=0, column=0, sticky='w')
        ttk.Entry(f, textvariable=self.batch_in_dir, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(f, text="Select", command=lambda: self.batch_in_dir.set(filedialog.askdirectory())).grid(row=0, column=2)
        
        # Output Directory
        ttk.Label(f, text="Save Folder:").grid(row=1, column=0, sticky='w', pady=10)
        ttk.Entry(f, textvariable=self.batch_out_dir, width=60).grid(row=1, column=1, padx=5)
        ttk.Button(f, text="Select", command=lambda: self.batch_out_dir.set(filedialog.askdirectory())).grid(row=1, column=2)
        
        # Options
        ttk.Checkbutton(f, text="Skip files already converted (matches filename_zimage.safetensors)", 
                        variable=self.skip_existing).grid(row=2, column=1, sticky='w')
        
        ttk.Button(f, text="Start Batch Conversion", command=self.run_batch).grid(row=3, column=1, pady=30)

    def browse_single_in(self):
        p = filedialog.askopenfilename(filetypes=[("LoRA", "*.safetensors *.pt")])
        if p:
            self.in_file.set(p)
            if not self.out_file.get():
                self.out_file.set(p.replace(".safetensors", "_zimage.safetensors").replace(".pt", "_zimage.safetensors"))

    def browse_single_out(self):
        p = filedialog.asksaveasfilename(defaultextension=".safetensors")
        if p: self.out_file.set(p)

    def run_single(self):
        def task():
            try:
                convert_lora(self.in_file.get(), self.out_file.get(), self.update_progress)
                self.log(f"Finished: {os.path.basename(self.out_file.get())}")
                messagebox.showinfo("Done", "Single conversion successful.")
            except Exception as e: self.log(f"Error: {e}")
        threading.Thread(target=task, daemon=True).start()

    def run_batch(self):
        def task():
            in_dir = self.batch_in_dir.get()
            out_dir = self.batch_out_dir.get() or in_dir
            
            if not os.path.isdir(in_dir):
                messagebox.showerror("Error", "Source folder not found.")
                return

            files = [f for f in os.listdir(in_dir) if f.endswith(('.safetensors', '.pt')) and "_zimage" not in f]
            if not files:
                self.log("No valid LoRA files found in source folder.")
                return
            
            self.log(f"Found {len(files)} files. Starting batch processing...")
            
            for i, filename in enumerate(files):
                in_p = os.path.join(in_dir, filename)
                # Create output filename
                name, ext = os.path.splitext(filename)
                out_p = os.path.join(out_dir, f"{name}_zimage.safetensors")
                
                # Check for existing
                if self.skip_existing.get() and os.path.exists(out_p):
                    self.log(f"Skipping (Already exists): {filename}")
                    continue
                
                try:
                    convert_lora(in_p, out_p)
                    self.log(f"[{i+1}/{len(files)}] Converted: {filename} -> {os.path.basename(out_p)}")
                except Exception as e:
                    self.log(f"Failed {filename}: {str(e)}")
                
                self.progress['value'] = ((i + 1) / len(files)) * 100
            
            self.log("Batch processing complete.")
            messagebox.showinfo("Done", "All files processed.")
        
        threading.Thread(target=task, daemon=True).start()

    def update_progress(self, val, status):
        self.progress['value'] = val
        if val % 20 == 0: self.log(status)

if __name__ == "__main__":
    app = LoRAConverterGUI()
    app.mainloop()