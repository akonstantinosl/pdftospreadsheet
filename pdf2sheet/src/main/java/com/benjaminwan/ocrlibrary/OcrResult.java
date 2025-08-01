package com.benjaminwan.ocrlibrary;

import java.util.ArrayList;

public class OcrResult implements OcrOutput {
    private final double dbNetTime;
    private final ArrayList<TextBlock> textBlocks;
    private final double detectTime;
    private final String strRes;

    public OcrResult(double dbNetTime, ArrayList<TextBlock> textBlocks, double detectTime, String strRes) {
        this.dbNetTime = dbNetTime;
        this.textBlocks = textBlocks;
        this.detectTime = detectTime;
        this.strRes = strRes;
    }

    public double getDbNetTime() {
        return dbNetTime;
    }

    public ArrayList<TextBlock> getTextBlocks() {
        return textBlocks;
    }

    public double getDetectTime() {
        return detectTime;
    }

    public String getStrRes() {
        return strRes;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("dbNetTime=%.2fms, detectTime=%.2fms\n", dbNetTime, detectTime));
        sb.append("textBlocks: \n");
        if (textBlocks != null) {
            for (TextBlock block : textBlocks) {
                sb.append(block.toString()).append("\n");
            }
        }
        sb.append("strRes: \n").append(strRes);
        return sb.toString();
    }
}