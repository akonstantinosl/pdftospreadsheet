package com.benjaminwan.ocrlibrary;

public class OcrEngine {

    static {
        try {
            System.loadLibrary("RapidOcrOnnx");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private int padding = 50;
    private float boxScoreThresh = 0.5f;
    private float boxThresh = 0.3f;
    private float unClipRatio = 1.6f;
    private boolean doAngle = true;
    private boolean mostAngle = true;

    // Getters
    public int getPadding() { return padding; }
    public float getBoxScoreThresh() { return boxScoreThresh; }
    public float getBoxThresh() { return boxThresh; }
    public float getUnClipRatio() { return unClipRatio; }
    public boolean isDoAngle() { return doAngle; }
    public boolean isMostAngle() { return mostAngle; }

    // Setters
    public void setPadding(int padding) { this.padding = padding; }
    public void setBoxScoreThresh(float boxScoreThresh) { this.boxScoreThresh = boxScoreThresh; }
    public void setBoxThresh(float boxThresh) { this.boxThresh = boxThresh; }
    public void setUnClipRatio(float unClipRatio) { this.unClipRatio = unClipRatio; }
    public void setDoAngle(boolean doAngle) { this.doAngle = doAngle; }
    public void setMostAngle(boolean mostAngle) { this.mostAngle = mostAngle; }

    /**
     * Overloaded detect method that uses the engine's properties.
     */
    public OcrResult detect(String input, int maxSideLen) {
        return detect(
                input, padding, maxSideLen,
                boxScoreThresh, boxThresh,
                unClipRatio, doAngle, mostAngle
        );
    }

    // External (Native) methods
    public native boolean setNumThread(int numThread);

    public native void initLogger(
            boolean isConsole,
            boolean isPartImg,
            boolean isResultImg
    );

    public native void enableResultText(String imagePath);

    public native boolean initModels(
            String modelsDir,
            String detName,
            String clsName,
            String recName,
            String keysName
    );

    public native String getVersion();

    /**
     * The core detection method that calls the native library.
     */
    public native OcrResult detect(
            String input, int padding, int maxSideLen,
            float boxScoreThresh, float boxThresh,
            float unClipRatio, boolean doAngle, boolean mostAngle
    );
}