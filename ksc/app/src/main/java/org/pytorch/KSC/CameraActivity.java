//https://github.com/Faisal-FS/CameraX-In-Java
//https://gist.github.com/moshimore/dfe5cf0216a520a8fef55ebe58a7ebe4
package org.pytorch.KSC;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.nio.ByteBuffer;
import java.util.Date;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;

public class CameraActivity extends AppCompatActivity implements ImageAnalysis.Analyzer, View.OnClickListener {
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

    PreviewView previewView;
    private ImageCapture imageCapture;
    private Button bCapture;
    private permission perm;
    private Bitmap finalBitmap;
    View crop;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        permissionCheck();
        previewView = findViewById(R.id.viewFinder);
        bCapture = findViewById(R.id.bCapture);
        crop = findViewById(R.id.crop);
        bCapture.setOnClickListener(new myListener() {
            @Override
            public void onSingleClick(View v) {
                imageCapture.takePicture(
                        getExecutor(),
                        new ImageCapture.OnImageCapturedCallback() {
                            @Override
                            public void onCaptureSuccess(ImageProxy capturedImage) {
                                Bitmap bitmap = imageProxyToBitmap(capturedImage);
                                Toast.makeText(CameraActivity.this, "Photo has been captured successfully.", Toast.LENGTH_SHORT).show();
                                preprocess(bitmap);
                            }

                            @Override
                            public void onError(@NonNull ImageCaptureException exception) {
                                Toast.makeText(CameraActivity.this, "Error capturing photo: " + exception.getMessage(), Toast.LENGTH_SHORT).show();
                            }
                        });
                //send();
            }
        });

        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                startCameraX(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, getExecutor());

    }

    private void permissionCheck(){
        perm = new permission(this,this);
        if(!perm.checkPermission()){
            perm.requestPermission();
        }

    }

    Executor getExecutor() {
        return ContextCompat.getMainExecutor(this);
    }

    @SuppressLint("RestrictedApi")
    private void startCameraX(ProcessCameraProvider cameraProvider) {
        cameraProvider.unbindAll();
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();
        Preview preview = new Preview.Builder()
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        // Image capture use case
        imageCapture = new ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build();

        // Image analysis use case
        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(getExecutor(), this);

        //bind to lifecycle:
        cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, preview, imageCapture);
    }

    @Override
    public void analyze(@NonNull ImageProxy image) {
        // image processing here for the current frame
        Log.d("TAG", "analyze: got the frame at: " + image.getImageInfo().getTimestamp());
        image.close();
    }

    @SuppressLint("RestrictedApi")
    @Override
    public void onClick(View view) {
    }
    private Bitmap imageProxyToBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy planeProxy = image.getPlanes()[0];
        ByteBuffer buffer = planeProxy.getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);

        return BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
    }
    //https://github.com/rrifafauzikomara/CustomCamera/tree/custom_camerax
    public void preprocess(Bitmap bitmap){
        int heightOriginal = previewView.getHeight();
        int widthOriginal = previewView.getWidth();
        int heightFrame = crop.getHeight();
        int widthFrame = crop.getWidth();
        int leftFrame = crop.getLeft();
        int topFrame = crop.getTop();
        int heightReal = bitmap.getHeight();
        int widthReal = bitmap.getWidth();
        int widthFinal = widthFrame * widthReal / widthOriginal;
        int heightFinal = heightFrame * heightReal / heightOriginal;
        int leftFinal = leftFrame * widthReal / widthOriginal;
        int topFinal = topFrame * heightReal / heightOriginal;
        widthFinal=widthFinal*7/10;
        leftFinal = leftFinal*12/9;
        Bitmap reprocessed = Bitmap.createBitmap(
                bitmap,
                leftFinal, topFinal, widthFinal, heightFinal
        );
        reprocessed = imgRotate(reprocessed);
        finalBitmap = Bitmap.createScaledBitmap(reprocessed,reprocessed.getWidth()/3,reprocessed.getHeight()/3,true);
                //
        Log.i("origin width",Integer.toString(bitmap.getWidth()));
        Log.i("origin height",Integer.toString(bitmap.getHeight()));
        Log.i("crop width",Integer.toString(reprocessed.getWidth()));
        Log.i("crop height",Integer.toString(reprocessed.getHeight()));
        Log.i("resized width",Integer.toString(finalBitmap.getWidth()));
        Log.i("resized height",Integer.toString(finalBitmap.getHeight()));
        send();
    }
    public void send(){
        Intent intent = new Intent(CameraActivity.this, classification.class);

        intent.putExtra("image", bitToByte(finalBitmap));
        startActivity(intent);
    }
    public static byte[] bitToByte(Bitmap bitmap){
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG,100,stream);
        byte[] byteArray = stream.toByteArray();
        return byteArray;
    }
    private Bitmap imgRotate(Bitmap bmp){
        int width = bmp.getWidth();
        int height = bmp.getHeight();
        Matrix matrix = new Matrix();
        matrix.postRotate(90);

        Bitmap resizedBitmap = Bitmap.createBitmap(bmp, 0, 0, width, height, matrix, true);
        bmp.recycle();

        return resizedBitmap;
    }


    //출처: https://gogorchg.tistory.com/entry/Android-Bitmap-이미지-가로-세로-회전 [항상 초심으로]
}


