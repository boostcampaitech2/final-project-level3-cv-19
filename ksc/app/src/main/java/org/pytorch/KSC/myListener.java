package org.pytorch.KSC;

import android.os.SystemClock;
import android.view.View;

public abstract class myListener implements View.OnClickListener{
    private static final long least_click_interval = 600;
    private long mLastClickTime = 0;

    public abstract void onSingleClick(View v);

    @Override
    public final void onClick(View v){
        long currentClickTime = SystemClock.uptimeMillis();
        long elapsedTime = currentClickTime - mLastClickTime;
        mLastClickTime = currentClickTime;

        if(elapsedTime > least_click_interval){
            onSingleClick(v);
        }
    }
}
