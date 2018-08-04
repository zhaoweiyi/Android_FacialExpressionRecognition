package com.example.alex.opencvdemo;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Color;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.widget.ImageView;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.widget.Button;
import android.view.View.OnClickListener;
import android.widget.TextView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.core.Rect;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.util.*;

public class MainActivity extends AppCompatActivity{
    static {
        if(!OpenCVLoader.initDebug())
        {
            Log.d("opencv","初始化失败");
        }
    }

    private ImageView imageView;
    private TextView reslutTextView;
    private Button getPhotoButton, greyPhotoButton, ferButton;
    private Classifier classifier;//识别类
    private static final String MODEL_FILE = "file:///android_asset/FacialExpressionReg.pb";
    private static final int IMAGE_SIZE = 48;
    private Uri imageUri;
    public static final int TAKE_PHOTO = 1;
    private CascadeClassifier cascadeClassifier = null; //级联分类器
    private int absoluteFaceSize = 0;

    public static Bitmap toGrayscale(Bitmap bmpOriginal) {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
         paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        return bmpGrayscale;
    }

    public static void putStringToTxt(String s, String name)
    {
        try
        {
            FileOutputStream outStream = new FileOutputStream("/sdcard/"+name+"cc.txt",true);
            OutputStreamWriter writer = new OutputStreamWriter(outStream,"gb2312");
            writer.write(s);
            writer.write("/n");
            writer.flush();
            writer.close();//记得关闭
            Log.d("hehe","wwwwwwwwwwwwww");
            outStream.close();
        }
        catch (Exception e)
        {
            Log.e("m", "file write error");
        }
    }

    //缩放图片,使用openCV，缩放方法采用area interpolation法
    private Bitmap scaleImage(Bitmap bitmap, int width, int height)
    {

        Mat src = new Mat();
        Mat dst = new Mat();
        Utils.bitmapToMat(bitmap, src);
        //new Size(width, height)
        Imgproc.resize(src, dst, new Size(width,height),0,0,Imgproc.INTER_AREA);
        Bitmap bitmap1 = Bitmap.createBitmap(dst.cols(),dst.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(dst, bitmap1);
        return bitmap1;
    }

    //Andorid不支持单通道图片，获取输入像素集
    public float[] getSingleChannelPixel(Bitmap bitmap) {
        float[] floatValues = new float[IMAGE_SIZE * IMAGE_SIZE * 1];

        if ((bitmap.getWidth() != IMAGE_SIZE) ||  (bitmap.getHeight() != IMAGE_SIZE)){
            Log.d("getSingleChannelPixel","获取像素时图片尺寸不对");
        }

        StringBuffer sBuffer = new StringBuffer("像素值：");
        for(int i = 0;i<bitmap.getWidth();i++)
        {
            for(int j =0;j<bitmap.getHeight();j++)
            {
                int col = bitmap.getPixel(i, j);
                int alpha = col&0xFF000000;
                int red = (col&0x00FF0000)>>16;
                int green = (col&0x0000FF00)>>8;
                int blue = (col&0x000000FF);
                int gray = (int)((float)red*0.3+(float)green*0.59+(float)blue*0.11);
                //int newColor = alpha|(gray<<16)|(gray<<8)|gray;
                floatValues[i + j* IMAGE_SIZE] = gray / 255.0f;
                sBuffer.append(gray) ;
                sBuffer.append(" ") ;

                if ((i == 0) && (j<10)) {
                    Log.d("hehe",j + ":" + String.valueOf(gray));
                }
                //Log.v("tag",  Integer.toHexString(col));
            }
        }

        putStringToTxt(sBuffer.toString(), "pixel");
        return floatValues;
    }

    Bitmap adjustPhotoRotation(Bitmap bm, final int orientationDegree)
    {
        Matrix m = new Matrix();
        m.setRotate(orientationDegree, (float) bm.getWidth() / 2, (float) bm.getHeight() / 2);

        try {
            Bitmap bm1 = Bitmap.createBitmap(bm, 0, 0, bm.getWidth(), bm.getHeight(), m, true);

            return bm1;

        } catch (OutOfMemoryError ex) {
        }
        return null;

    }

    private void detectFace(Bitmap bitmap)
    {

        Mat img = new Mat();
        Utils.bitmapToMat(bitmap, img);

        Mat imgGray = new Mat();;
        MatOfRect faces = new MatOfRect();

        if(img.empty())
        {
            Log.d("ccx","detectFace but img is empty");
            return;
        }

        if(img.channels() ==3)
        {
            Imgproc.cvtColor(img, imgGray, Imgproc.COLOR_RGB2GRAY);      // RGB转化为灰度
        }
        else
        {
            imgGray = img;      // 不转化
        }

        cascadeClassifier.detectMultiScale(imgGray, faces, 1.1, 2, 2, new Size(absoluteFaceSize, absoluteFaceSize), new Size());

        Rect[] facesArray = faces.toArray();
        if (facesArray.length > 0){
            for (int i = 0; i < facesArray.length; i++) {    //用框标记
                Imgproc.rectangle(imgGray, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
                Log.d("ccx","index:" + i + "topLeft:" + facesArray[i].tl() + "bottomRight:" + facesArray[i].br()+ "height:" + facesArray[i].height);
            }
        }

        Utils.matToBitmap(imgGray, bitmap);
        imageView.setImageBitmap(bitmap);
        Bitmap destBitmap = Bitmap.createBitmap(bitmap, (int) (facesArray[0].tl().x), (int) (facesArray[0].tl().y), facesArray[0].width, facesArray[0].height);
        Bitmap scaleImage = scaleImage(destBitmap, 48, 48);
        Bitmap bitmap5 = toGrayscale(scaleImage);
        Bitmap bitmap6 = adjustPhotoRotation(bitmap5, 270);

        classifier = new Classifier(getAssets(),MODEL_FILE);
        ArrayList<String> result = classifier.predict(getSingleChannelPixel(bitmap6));
        //0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
        String str = result.get(0);
        switch(str){
            case "0":
                str = "生气";break;
            case "1":
                str = "厌恶";break;
            case "2":
                str = "恐惧";break;
            case "3":
                str = "开心";break;
            case "4":
                str = "难过";break;
            case "5":
                str = "惊讶";break;
            case "6":
                str = "平静";break;
            default:
                Log.d("ccx","Tensorflow return is error.");;break;
        }
        reslutTextView.setText("识别结果: " + str);

        return;
    }

    private void initializeOpenCVDependencies() {
        try {
            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface_improved); //OpenCV的人脸模型文件： lbpcascade_frontalface_improved
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface_improved.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            // 加载cascadeClassifier
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e("opencv","Error loading cascade");
        }
    }

    class MyClickListener implements OnClickListener{

        @Override
        public void onClick(View v) {
            // TODO Auto-generated method stub
            switch (v.getId()) {
                case R.id.btn_get_photo:
                    File outputImage = new File(Environment.getExternalStorageDirectory(),
                            "tempImage" + ".jpg");
                    try {
                        if (outputImage.exists()) {
                            outputImage.delete();
                        }
                        outputImage.createNewFile();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    imageUri = Uri.fromFile(outputImage);
                    Intent intent = new Intent("android.media.action.IMAGE_CAPTURE");
                    intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
                    startActivityForResult(intent, 0);
                    break;
                case R.id.btn_grey_photo:
                    //Bitmap bitmap2 = RGBtoGrayscale();
                    // Bitmap bitmap = BitmapFactory.decodeResource(MainActivity.this.getApplicationContext().getResources(), R.drawable.gqsz);
                    //Bitmap bitmap1 = scaleImage(bitmap, 48, 48);
                    //Bitmap bitmap2 = toGrayscale(bitmap1);
                    String pathString = Environment.getExternalStorageDirectory() + "/tempImage" + ".jpg";
                    Log.d("matrix",pathString + "");
                    Bitmap bitmap = null;
                    try
                    {
                        File file = new File(pathString);
                        if(file.exists())
                        {
                            bitmap = BitmapFactory.decodeFile(pathString);
                        }
                    } catch (Exception e)
                    {
                        // TODO: handle exception
                    }
                    detectFace(bitmap);
                    break;
                default:
                    break;
            }
        }

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == 0) {
            imageView.setImageURI(imageUri);
        }
     }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getSupportActionBar().hide();
        setContentView(R.layout.activity_main);
        imageView = (ImageView)findViewById(R.id.img_ccx);
        getPhotoButton = (Button)findViewById(R.id.btn_get_photo);
        greyPhotoButton = (Button)findViewById(R.id.btn_grey_photo);
        reslutTextView = (TextView)findViewById(R.id.tv_output);

        initializeOpenCVDependencies();

        getPhotoButton.setOnClickListener(new MyClickListener());
        greyPhotoButton.setOnClickListener(new MyClickListener());
    }
}