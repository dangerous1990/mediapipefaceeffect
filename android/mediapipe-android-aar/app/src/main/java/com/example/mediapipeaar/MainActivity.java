// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.example.mediapipeaar;

import android.content.pm.ApplicationInfo;
import android.graphics.Color;
import android.graphics.SurfaceTexture;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.GestureDetector;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewGroup.LayoutParams;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.content.pm.PackageManager;
import android.content.pm.PackageManager.NameNotFoundException;

import androidx.appcompat.app.AppCompatActivity;

import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.CameraXPreviewHelper;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.FrameProcessor;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.glutil.EglManager;
import com.google.mediapipe.modules.facegeometry.FaceGeometryProto.FaceGeometry;
import com.google.mediapipe.formats.proto.MatrixDataProto.MatrixData;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Main activity of MediaPipe face mesh app. */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";

    static {
        // Load all native libraries needed by the app.
        try {
            System.loadLibrary("mediapipe_jni");
            System.loadLibrary("opencv_java3");
        }catch (Exception e) {
            Log.e(TAG, "load jni package failed: " + e);
        }

    }
    private static final String BINARY_GRAPH_NAME = "face_effect_gpu.binarypb";
    private static final String INPUT_VIDEO_STREAM_NAME = "input_video";
    private static final String OUTPUT_VIDEO_STREAM_NAME = "output_video";

    // Flips the camera-preview frames vertically before sending them into FrameProcessor to be
    // processed in a MediaPipe graph, and flips the processed frames back when they are displayed.
    // This is needed because OpenGL represents images assuming the image origin is at the bottom-left
    // corner, whereas MediaPipe in general assumes the image origin is at top-left.
    private static final boolean FLIP_FRAMES_VERTICALLY = true;


    // Side packet / stream names.
    private static final String USE_FACE_DETECTION_INPUT_SOURCE_INPUT_SIDE_PACKET_NAME =
            "use_face_detection_input_source";
    private static final String SELECTED_EFFECT_ID_INPUT_STREAM_NAME = "selected_effect_id";
    private static final String OUTPUT_FACE_GEOMETRY_STREAM_NAME = "multi_face_geometry";

    private static final String EFFECT_SWITCHING_HINT_TEXT = "Tap to switch between effects!";

    private static final boolean USE_FACE_DETECTION_INPUT_SOURCE = false;
    private static final int MATRIX_TRANSLATION_Z_INDEX = 14;

    private static final int SELECTED_EFFECT_ID_AXIS = 0;
    private static final int SELECTED_EFFECT_ID_FACEPAINT = 1;
    private static final int SELECTED_EFFECT_ID_GLASSES = 2;

    private final Object effectSelectionLock = new Object();
    private int selectedEffectId;

    private View effectSwitchingHintView;
    private GestureDetector tapGestureDetector;

    // {@link SurfaceTexture} where the camera-preview frames can be accessed.
    private SurfaceTexture previewFrameTexture;
    // {@link SurfaceView} that displays the camera-preview frames processed by a MediaPipe graph.
    private SurfaceView previewDisplayView;
    // Creates and manages an {@link EGLContext}.
    private EglManager eglManager;
    // Sends camera-preview frames into a MediaPipe graph for processing, and displays the processed
    // frames onto a {@link Surface}.
    private FrameProcessor processor;
    // Converts the GL_TEXTURE_EXTERNAL_OES texture from Android camera into a regular texture to be
    // consumed by {@link FrameProcessor} and the underlying MediaPipe graph.
    private ExternalTextureConverter converter;
    // ApplicationInfo for retrieving metadata defined in the manifest.
    private ApplicationInfo applicationInfo;
    // Handles camera access via the {@link CameraX} Jetpack support library.
    private CameraXPreviewHelper cameraHelper;
    // Used to obtain the content view for this application. If you are extending this class, and
    // have a custom layout, override this method and return the custom layout.
    protected int getContentViewLayoutResId() {
        return R.layout.activity_main;
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(getContentViewLayoutResId());

        try {
            applicationInfo =
                    getPackageManager().getApplicationInfo(getPackageName(), PackageManager.GET_META_DATA);
        } catch (NameNotFoundException e) {
            Log.e(TAG, "Cannot find application info: " + e);
        }

        previewDisplayView = new SurfaceView(this);
        setupPreviewDisplayView();

        AndroidAssetUtil.initializeNativeAssetManager(this);
        eglManager = new EglManager(null);
        processor =
                new FrameProcessor(
                        this,
                        eglManager.getNativeContext(),
                        BINARY_GRAPH_NAME,
                        INPUT_VIDEO_STREAM_NAME,
                        OUTPUT_VIDEO_STREAM_NAME);
        processor
                .getVideoSurfaceOutput()
                .setFlipY(FLIP_FRAMES_VERTICALLY);
        PermissionHelper.checkAndRequestCameraPermissions(this);


        // Add an effect switching hint view to the preview layout.
        effectSwitchingHintView = createEffectSwitchingHintView();
        effectSwitchingHintView.setVisibility(View.INVISIBLE);
        ViewGroup viewGroup = findViewById(R.id.preview_display_layout);
        viewGroup.addView(effectSwitchingHintView);

        // By default, render the axis effect for the face detection input source and the glasses effect
        // for the face landmark input source.
        if (USE_FACE_DETECTION_INPUT_SOURCE) {
            selectedEffectId = SELECTED_EFFECT_ID_AXIS;
        } else {
            selectedEffectId = SELECTED_EFFECT_ID_GLASSES;
        }

        // Pass the USE_FACE_DETECTION_INPUT_SOURCE flag value as an input side packet into the graph.
        Map<String, Packet> inputSidePackets = new HashMap<>();
        inputSidePackets.put(
                USE_FACE_DETECTION_INPUT_SOURCE_INPUT_SIDE_PACKET_NAME,
                processor.getPacketCreator().createBool(USE_FACE_DETECTION_INPUT_SOURCE));
        processor.setInputSidePackets(inputSidePackets);

        // This callback demonstrates how the output face geometry packet can be obtained and used
        // in an Android app. As an example, the Z-translation component of the face pose transform
        // matrix is logged for each face being equal to the approximate distance away from the camera
        // in centimeters.
        processor.addPacketCallback(
                OUTPUT_FACE_GEOMETRY_STREAM_NAME,
                (packet) -> {
                    effectSwitchingHintView.post(
                            () ->
                                    effectSwitchingHintView.setVisibility(
                                            USE_FACE_DETECTION_INPUT_SOURCE ? View.INVISIBLE : View.VISIBLE));

                    Log.d(TAG, "Received a multi face geometry packet.");
                    List<FaceGeometry> multiFaceGeometry =
                            PacketGetter.getProtoVector(packet, FaceGeometry.parser());

                    StringBuilder approxDistanceAwayFromCameraLogMessage = new StringBuilder();
                    for (FaceGeometry faceGeometry : multiFaceGeometry) {
                        if (approxDistanceAwayFromCameraLogMessage.length() > 0) {
                            approxDistanceAwayFromCameraLogMessage.append(' ');
                        }
                        MatrixData poseTransformMatrix = faceGeometry.getPoseTransformMatrix();
                        approxDistanceAwayFromCameraLogMessage.append(
                                -poseTransformMatrix.getPackedData(MATRIX_TRANSLATION_Z_INDEX));
                    }

                    Log.d(
                            TAG,
                            "[TS:"
                                    + packet.getTimestamp()
                                    + "] size = "
                                    + multiFaceGeometry.size()
                                    + "; approx. distance away from camera in cm for faces = ["
                                    + approxDistanceAwayFromCameraLogMessage
                                    + "]");
                });

        // Alongside the input camera frame, we also send the `selected_effect_id` int32 packet to
        // indicate which effect should be rendered on this frame.
        processor.setOnWillAddFrameListener(
                (timestamp) -> {
                    Packet selectedEffectIdPacket = null;
                    try {
                        synchronized (effectSelectionLock) {
                            selectedEffectIdPacket = processor.getPacketCreator().createInt32(selectedEffectId);
                        }

                        processor
                                .getGraph()
                                .addPacketToInputStream(
                                        SELECTED_EFFECT_ID_INPUT_STREAM_NAME, selectedEffectIdPacket, timestamp);
                    } catch (RuntimeException e) {
                        Log.e(
                                TAG, "Exception while adding packet to input stream while switching effects: " + e);
                    } finally {
                        if (selectedEffectIdPacket != null) {
                            selectedEffectIdPacket.release();
                        }
                    }
                });

        // We use the tap gesture detector to switch between face effects. This allows users to try
        // multiple pre-bundled face effects without a need to recompile the app.
        tapGestureDetector =
                new GestureDetector(
                        this,
                        new GestureDetector.SimpleOnGestureListener() {
                            @Override
                            public void onLongPress(MotionEvent event) {
                                switchEffect();
                            }

                            @Override
                            public boolean onSingleTapUp(MotionEvent event) {
                                switchEffect();
                                return true;
                            }

                            private void switchEffect() {
                                // Avoid switching the Axis effect for the face detection input source.
                                if (USE_FACE_DETECTION_INPUT_SOURCE) {
                                    return;
                                }

                                // Looped effect order: glasses -> facepaint -> axis -> glasses -> ...
                                synchronized (effectSelectionLock) {
                                    switch (selectedEffectId) {
                                        case SELECTED_EFFECT_ID_AXIS:
                                        {
                                            selectedEffectId = SELECTED_EFFECT_ID_GLASSES;
                                            break;
                                        }

                                        case SELECTED_EFFECT_ID_FACEPAINT:
                                        {
                                            selectedEffectId = SELECTED_EFFECT_ID_AXIS;
                                            break;
                                        }

                                        case SELECTED_EFFECT_ID_GLASSES:
                                        {
                                            selectedEffectId = SELECTED_EFFECT_ID_FACEPAINT;
                                            break;
                                        }

                                        default:
                                            break;
                                    }
                                }
                            }
                        });
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        return tapGestureDetector.onTouchEvent(event);
    }

    private View createEffectSwitchingHintView() {
        TextView effectSwitchingHintView = new TextView(getApplicationContext());
        effectSwitchingHintView.setLayoutParams(
                new RelativeLayout.LayoutParams(LayoutParams.FILL_PARENT, LayoutParams.FILL_PARENT));
        effectSwitchingHintView.setText(EFFECT_SWITCHING_HINT_TEXT);
        effectSwitchingHintView.setGravity(Gravity.CENTER_HORIZONTAL | Gravity.BOTTOM);
        effectSwitchingHintView.setPadding(0, 0, 0, 480);
        effectSwitchingHintView.setTextColor(Color.parseColor("#ffffff"));
        effectSwitchingHintView.setTextSize((float) 24);

        return effectSwitchingHintView;
    }

    @Override
    protected void onResume() {
        super.onResume();
        converter =
                new ExternalTextureConverter(
                        eglManager.getContext(), 2);
        converter.setFlipY(FLIP_FRAMES_VERTICALLY);
        converter.setConsumer(processor);
        if (PermissionHelper.cameraPermissionsGranted(this)) {
            startCamera();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        converter.close();

        // Hide preview display until we re-open the camera again.
        previewDisplayView.setVisibility(View.GONE);
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    protected void onCameraStarted(SurfaceTexture surfaceTexture) {
        previewFrameTexture = surfaceTexture;
        // Make the display view visible to start showing the preview. This triggers the
        // SurfaceHolder.Callback added to (the holder of) previewDisplayView.
        previewDisplayView.setVisibility(View.VISIBLE);
    }

    protected Size cameraTargetResolution() {
        return null; // No preference and let the camera (helper) decide.
    }

    public void startCamera() {
        cameraHelper = new CameraXPreviewHelper();
        cameraHelper.setOnCameraStartedListener(
                surfaceTexture -> {
                    onCameraStarted(surfaceTexture);
                });
        CameraHelper.CameraFacing cameraFacing = CameraHelper.CameraFacing.FRONT;
        cameraHelper.startCamera(
                this, cameraFacing, /*unusedSurfaceTexture=*/ null, cameraTargetResolution());
    }

    protected Size computeViewSize(int width, int height) {
        return new Size(width, height);
    }

    protected void onPreviewDisplaySurfaceChanged(
            SurfaceHolder holder, int format, int width, int height) {
        // (Re-)Compute the ideal size of the camera-preview display (the area that the
        // camera-preview frames get rendered onto, potentially with scaling and rotation)
        // based on the size of the SurfaceView that contains the display.
        Size viewSize = computeViewSize(width, height);
        Size displaySize = cameraHelper.computeDisplaySizeFromViewSize(viewSize);
        boolean isCameraRotated = cameraHelper.isCameraRotated();

        // Connect the converter to the camera-preview frames as its input (via
        // previewFrameTexture), and configure the output width and height as the computed
        // display size.
        converter.setSurfaceTextureAndAttachToGLContext(
                previewFrameTexture,
                isCameraRotated ? displaySize.getHeight() : displaySize.getWidth(),
                isCameraRotated ? displaySize.getWidth() : displaySize.getHeight());
    }

    private void setupPreviewDisplayView() {
        previewDisplayView.setVisibility(View.GONE);
        ViewGroup viewGroup = findViewById(R.id.preview_display_layout);
        viewGroup.addView(previewDisplayView);

        previewDisplayView
                .getHolder()
                .addCallback(
                        new SurfaceHolder.Callback() {
                            @Override
                            public void surfaceCreated(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(holder.getSurface());
                            }

                            @Override
                            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                                onPreviewDisplaySurfaceChanged(holder, format, width, height);
                            }

                            @Override
                            public void surfaceDestroyed(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(null);
                            }
                        });
    }

}
