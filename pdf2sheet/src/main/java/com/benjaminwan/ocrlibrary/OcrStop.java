package com.benjaminwan.ocrlibrary;

/**
 * Represents a stopped OCR operation state. Implemented as a singleton.
 */
public final class OcrStop implements OcrOutput {
    public static final OcrStop INSTANCE = new OcrStop();

    private OcrStop() {
    }
}