# Lab Assignment 9 — Vision-Based Cognitive Assistants

Includes **both** proposed projects from the assignment PDF.

## Medicine bottle label reader

Install Tesseract OCR on your computer (macOS: `brew install tesseract`) and the Python dependencies in `requirements.txt`. Run `python medicine_label_reader.py your_image.jpg` on a label photo you have permission to use. The program extracts printed text and potential expiry-date strings, but is not a medication safety or dosage decision tool. Always check the physical label.

## Facial expression assistant

Install `deepface` separately (`pip install deepface`) and run `python facial_expression_assistant.py permitted_face_image.jpg`. This is an experimental expression classifier. It cannot accurately determine internal feelings or mental health; use only with the subject's consent.

The assignment mentions Landing AI, but no credentials or starter files were provided. These scripts are local Python demonstrations of the same tasks, not an integration with Landing AI.
