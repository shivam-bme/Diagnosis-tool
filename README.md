# Web-Based Diagnosis Tool for Medical Imaging

### Overview
This repository features a web-based diagnostic tool designed to aid in medical imaging analysis, with a focus on lung, liver, bone fractures, and tumors.

### Features
- **MRI and X-Ray Analysis:** Supports multiple imaging types with specific diagnostic web pages.
- **Automated Image Processing:** Includes image conversion and model prediction workflows.
- **Interactive UI:** HTML interfaces for each diagnostic module.

### Files
- **app.js, server.js:** Backend setup for running the web server and handling requests.
- **HTML files:** Individual pages for each diagnosis type (e.g., `lung.html`, `bone.html`).
- **tfjs_model, models:** Pre-trained models for medical imaging.
- **convert_mha_to_png.py:** Script for image format conversion.

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/shivam-bme/Diagnosis-tool.git
cd Diagnosis-tool
npm install
