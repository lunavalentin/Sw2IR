# Sw2IR (Sweep-to-IR) 🌈🦄

A magical little app to batch deconvolve your audio sweeps into Impulse Responses (IRs).

## Features
- **Batch Processing**: Drag and drop 100s of files.
- **Auto-Alignment**: Optionally aligns the direct sound to 10ms.
- **Metadata**: Preserves scalefactor metadata for Pyrat2/IRIS.
- **Rainbows**: Because why not?

## Installation

1. Open Terminal.
2. Install Python (if you don't have it).
3. Install the dependencies:
   ```bash
   pip3 install PySide6 soundfile scipy numpy
   ```

## How to Run

1. Open Terminal in this folder.
2. Run:
   ```bash
   python3 Sw2IR.py
   ```

## Usage

1. Drag your **Original Sweep (Reference)** into Box 1.
2. Choose a folder to save the IRs in Box 2.
3. Drag your **Measurement Sweeps** into Box 3.
4. Hit **MAKE IT HAPPEN!** ✨

Enjoy!
