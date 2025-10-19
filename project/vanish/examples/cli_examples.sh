#!/bin/bash
# Vanish CLI Usage Examples

# Basic usage
echo "Example 1: Basic usage"
vanish input.wav -o output.wav

# With verbose output
echo -e "\nExample 2: Verbose output"
vanish input.wav -o output.wav -vv

# High quality mode
echo -e "\nExample 3: High quality processing"
vanish input.wav -o output.wav --quality high

# Fast processing mode
echo -e "\nExample 4: Fast processing"
vanish input.wav -o output.wav --quality fast

# GPU processing
echo -e "\nExample 5: Force GPU processing"
vanish input.wav -o output.wav --device cuda

# CPU processing
echo -e "\nExample 6: CPU processing (slower)"
vanish input.wav -o output.wav --device cpu

# Use VoiceFixer instead of Resemble-Enhance
echo -e "\nExample 7: Use VoiceFixer enhancement"
vanish input.wav -o output.wav --enhancement voicefixer

# Save intermediate files
echo -e "\nExample 8: Save intermediate processing steps"
vanish input.wav -o output.wav --save-intermediate

# Use custom configuration
echo -e "\nExample 9: Custom configuration file"
vanish input.wav -o output.wav --config config.yaml

# Batch processing
echo -e "\nExample 10: Batch process directory"
vanish batch ./inputs ./outputs --pattern "*.wav" --prefix "clean_"

# Create configuration file
echo -e "\nExample 11: Generate configuration file"
vanish create-config my_config.yaml --preset rtx3060

# System information
echo -e "\nExample 12: Display system info"
vanish info

# Complete example with all options
echo -e "\nExample 13: Complete example"
vanish input.wav \
    -o output.wav \
    --config config.yaml \
    --device cuda \
    --quality high \
    --enhancement resemble \
    --metrics \
    --save-intermediate \
    -vv
