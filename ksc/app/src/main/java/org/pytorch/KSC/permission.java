package org.pytorch.KSC;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class permission {

    private Context context;
    private Activity activity;

    private String[] permissions ={
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.CAMERA
    };

    public permission(Activity _activity, Context _context){
        this.activity = _activity;
        this.context = _context;
    }

    private final int perm_num = 3;

    public boolean checkPermission(){
        for (String pm : permissions){
            if (ContextCompat.checkSelfPermission(context,pm)!= PackageManager.PERMISSION_GRANTED){
                return false;
            }
        }
        return true;
    }

    public void requestPermission(){
        ActivityCompat.requestPermissions(activity,permissions,perm_num);
    }
}
