
# Privacy Preserving Deepfake Detection System

## Overview
This project focuses on building a deepfake detection system that emphasizes privacy preservation through the use of Fully Homomorphic Encryption (FHE). The system allows for deepfake detection on encrypted data, ensuring that sensitive information remains private throughout the process.

<img width="1513" alt="image" src="https://github.com/user-attachments/assets/0ec21ce1-8b20-4b74-ad9e-547bad7c9327">

## Key Features
- **Deepfake Detection**: Machine learning model with 96% accuracy.
- **Privacy Focused**: Fully Homomorphic Encryption (FHE) allows computations on encrypted data, maintaining full user privacy.
- **GDPR Compliance**: The system is fully compliant with GDPR and other privacy-related regulations.
- **Scalable and Future Ready**: Designed to support image detection now, with plans for extending to real-time video deepfake detection.

## Technical Specifications
- **Model Accuracy**: 96% on current testing.
- **Model Size**: 11.177.538
- **Encryption Method**: Fully Homomorphic Encryption (FHE), ensuring no data is decrypted during processing.
- **Performance**:
  - Current processing time: ~1 hour per image.
  - Future optimization goal: <1 second per image by 2025.
- **Hardware**: Uses 6,172,672 Programmable Bootstrap Units.
- **Data Privacy**: User data is encrypted end-to-end, and users do not have access to the internal model.
  
## System Architecture
- The architecture is built around secure, private processing of media inputs through FHE.
- The current system supports image deepfake detection, and will later support video input.
- Full user data privacy is ensured by not decrypting the data at any stage of the analysis.

## Future Work
- Extend deepfake detection to handle real-time video streams.
- Optimize processing times to <1 second per image.
- Train the model with larger datasets for improved accuracy.
- Integrate with video conferencing platforms for live deepfake detection.

## Usage Instructions
1. **Input**: Upload an encrypted image or video for analysis.
2. **Detection Process**: The system processes the media through the encrypted deepfake detection model.
3. **Output**: Returns whether the media is likely a deepfake, without exposing any sensitive data.
