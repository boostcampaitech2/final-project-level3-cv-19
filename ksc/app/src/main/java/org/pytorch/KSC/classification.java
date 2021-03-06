package org.pytorch.KSC;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.text.DecimalFormat;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.MemoryFormat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import androidx.appcompat.app.AppCompatActivity;

public class classification extends AppCompatActivity {
    private Button toCamera;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        byte[] byteArray = getIntent().getByteArrayExtra("image");
        Bitmap capturedBitmap = BitmapFactory.decodeByteArray(byteArray,0,byteArray.length);
        setContentView(R.layout.activity_classification);
        toCamera = findViewById(R.id.camera_continue);
        toCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(classification.this, CameraActivity.class);
                startActivity(intent); // go to camera activity
            }
        });
        Module module = null;
        try {
            // loading serialized torchscript module from packaged into app android asset model.pt,
            // app/src/model/assets/model.pt
            module = LiteModuleLoader.load(assetFilePath(this, "model_resnet50.ptl"));
        } catch (IOException e) {
            Log.e("PytorchHelloWorld", "Error reading assets", e);
            finish();
        }

        // showing image on UI
        ImageView imageView = findViewById(R.id.class_image);
        imageView.setImageBitmap(capturedBitmap);
        detect(capturedBitmap,module);
    }

    public void detect(Bitmap bitmap, Module module){

        // preparing input tensor
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);

        // running the model
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        // getting tensor content as java array of floats
        final float[] scores = outputTensor.getDataAsFloatArray();

        // searching for the index with maximum score
        float maxScore = -Float.MAX_VALUE;
        float secondScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        int secondScoreIdx = -1;
        float sum = 0;
        for (int i = 0; i < scores.length; i++) {
            sum+=Math.exp(scores[i]);
            if (scores[i] > maxScore) {
                if(maxScore > secondScore){
                    secondScore = maxScore;
                    secondScoreIdx = maxScoreIdx;
                }
                maxScore = scores[i];
                maxScoreIdx = i;
            }
            else if(scores[i] > secondScore){
                secondScore = scores[i];
                secondScoreIdx = i;
            }
        }
        float maxpercent = (float)Math.exp(maxScore) / sum * 100;
        float secondpercent = (float)Math.exp(secondScore) / sum * 100;
        String firstName = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];
        String secondName = ImageNetClasses.IMAGENET_CLASSES[secondScoreIdx];
        String score1 = new DecimalFormat("#.00").format (maxpercent);
        String score2 = new DecimalFormat("#.00").format (secondpercent);
        // showing className on UI
        TextView textView = findViewById(R.id.first_class);
        textView.setText(firstName);
        TextView scoreView = findViewById(R.id.first_score);
        scoreView.setText(score1+ '%');
        TextView textView2 = findViewById(R.id.second_class);
        textView2.setText(secondName);
        TextView scoreView2 = findViewById(R.id.second_score);
        scoreView2.setText(score2+ '%');
    }

    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}
