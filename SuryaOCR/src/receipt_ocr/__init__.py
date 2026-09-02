"""Receipt OCR pipeline built on Surya OCR.

Deliberately does not eagerly import `receipt_ocr.pipeline` here: that pulls
in torch/surya, which is slow and unnecessary for code (e.g. tests) that only
needs lightweight modules like `receipt_ocr.reconstruct`. Import what you
need directly, e.g.:

    from receipt_ocr.pipeline import process_image, process_batch
"""
