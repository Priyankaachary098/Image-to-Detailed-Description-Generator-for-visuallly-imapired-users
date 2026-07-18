# Blind Assistance System using AI

An AI-powered assistive application designed to help visually impaired individuals better understand their surroundings. The system combines computer vision, image captioning, optical character recognition (OCR), and text-to-speech technologies to provide real-time audio descriptions of the environment.

## Features

- Detects and identifies multiple objects in an image using YOLOv8.
- Generates natural language image descriptions using the BLIP image captioning model.
- Extracts readable text from images using EasyOCR.
- Converts generated descriptions and detected text into speech using Google Text-to-Speech (gTTS).
- Simple and interactive web interface built with Gradio.
- Provides an end-to-end AI solution for environmental awareness.

## Technologies Used

- Python
- PyTorch
- YOLOv8 (Ultralytics)
- OpenCV
- BLIP Image Captioning
- EasyOCR
- Google Text-to-Speech (gTTS)
- Gradio

## Project Workflow

1. Upload an image through the Gradio interface.
2. Detect objects using YOLOv8.
3. Generate a detailed scene description with BLIP.
4. Extract visible text using EasyOCR.
5. Convert the generated description into speech.
6. Display results and provide downloadable audio output.

## Applications

- Assistive technology for visually impaired users
- Smart navigation assistance
- AI-powered accessibility solutions
- Computer Vision and Deep Learning research
- Educational demonstration of multimodal AI

## Future Enhancements

- Real-time webcam support
- Currency and document recognition
- Traffic signal and pedestrian crossing detection
- Multi-language voice output
- Mobile application integration
- Offline inference support

## Installation

```bash
git clone https://github.com/your-username/blind-assistance-system.git
cd blind-assistance-system
pip install -r requirements.txt
python app.py
```

## Results

The system successfully integrates object detection, image captioning, OCR, and speech synthesis to generate meaningful audio descriptions, making visual information more accessible to users with visual impairments.

## Author

**Priyanka P**

If you found this project useful, consider giving it a ⭐ on GitHub.
