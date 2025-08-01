package com.benjaminwan.ocrlibrary;

/**
 * Represents a failed OCR operation state. Implemented as a singleton.
 */
public final class OcrFailed implements OcrOutput {
    public static final OcrFailed INSTANCE = new OcrFailed();

    private OcrFailed() {
    }
}