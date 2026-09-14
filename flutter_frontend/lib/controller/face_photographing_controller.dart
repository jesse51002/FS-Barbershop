import 'dart:convert';
import 'dart:io';
import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:get/get.dart';
import 'package:hair_style_app/confing/app_colors.dart';
import 'package:hair_style_app/confing/end_points.dart';
import 'package:hair_style_app/controller/home_controller.dart';
import 'package:hair_style_app/controller/transformation_result_controller.dart';
import 'package:hair_style_app/http_handler/http_handler.dart';
import 'package:hair_style_app/screens/face_photographing/face_photographing_screen.dart';
import 'package:hair_style_app/screens/transformation_result/transformation_result_screen.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:image_picker/image_picker.dart';
import '../model/get_rendering_result.dart';

class FacePhotoGraphingController extends GetxController {
  CarouselController buttonCarouselController = CarouselController();
  bool isLoading = false;
  bool isInProgress = false;
  int tempIndex = 0;
  bool isEnable = false;
  // File? imageFile;
  GetRenderingResultModel? getRenderingResultModel;

  final myImagesList = [];

  Future<File?> pickImage() async {
    try {
      final image = await ImagePicker().pickImage(source: ImageSource.camera);
      ImageCropper croppedFile = ImageCropper();
      if (image == null) {
        return null;
      } else {
        final imageTemp = File(image.path);

        final imageData = await croppedFile.cropImage(
          sourcePath: imageTemp.path,
          aspectRatioPresets: [
            CropAspectRatioPreset.square,
            CropAspectRatioPreset.ratio3x2,
            CropAspectRatioPreset.original,
            CropAspectRatioPreset.ratio4x3,
            CropAspectRatioPreset.ratio16x9,
          ],
          uiSettings: [
            AndroidUiSettings(
                toolbarTitle: 'Cropper',
                toolbarColor: AppColors.primaryColor,
                toolbarWidgetColor: Colors.white,
                statusBarColor: AppColors.primaryColor,
                initAspectRatio: CropAspectRatioPreset.original,
                lockAspectRatio: false),
            IOSUiSettings(
              title: 'Cropper',
            ),
          ],
        );
        if (imageData != null) {
          File? imageFile = File(imageData.path);
          isEnable = true;
          myImagesList.add(imageFile);
          update();
          if (tempIndex < (myImagesList.length - 1)) {
            buttonCarouselController.nextPage();
            tempIndex = tempIndex + 1;
            update();
          }
          return imageFile;
        }
      }
    } on PlatformException catch (e) {
      print('Failed to pick image: $e');
    }

    update();
    return null;
  }

  Future<void> replaceImage(int index) async {
    try {
      final image = await ImagePicker().pickImage(source: ImageSource.camera);
      ImageCropper croppedFile = ImageCropper();
      if (image == null) {
        return;
      } else {
        final imageTemp = File(image.path);

        final imageData = await croppedFile.cropImage(
          sourcePath: imageTemp.path,
          aspectRatioPresets: [
            CropAspectRatioPreset.square,
            CropAspectRatioPreset.ratio3x2,
            CropAspectRatioPreset.original,
            CropAspectRatioPreset.ratio4x3,
            CropAspectRatioPreset.ratio16x9,
          ],
          uiSettings: [
            AndroidUiSettings(
                toolbarTitle: 'Cropper',
                toolbarColor: AppColors.primaryColor,
                toolbarWidgetColor: Colors.white,
                statusBarColor: AppColors.primaryColor,
                initAspectRatio: CropAspectRatioPreset.original,
                lockAspectRatio: false),
            IOSUiSettings(
              title: 'Cropper',
            ),
          ],
        );

        if (imageData != null) {
          File? imageFile = File(imageData.path);
          isEnable = true;
          myImagesList[index] =
              imageFile; // Replace the image at the specified index
          update();
        }
      }
    } on PlatformException catch (e) {
      print('Failed to pick image: $e');
    }
  }

  getImageId() async {
    isLoading = true;
    update();
    final eventId = getStorage.read("eventId");

    await HttpHandler.getHttpMethod(
            url: EndPoints().getImageId(eventId: eventId))
        .then((value) {
      if (value["statusCode"] == 200) {
        final response = jsonDecode(value["body"]);
        isInProgress = response["inprogress"];
        if (isInProgress) {
          Fluttertoast.showToast(msg: "Waiting for image to be uploaded…");
        } else {
          Get.find<HomeController>().selectedImageId = response["hairstyle_id"];
          Get.to(const FacePhotoGraphingScreen());
          Fluttertoast.showToast(
              msg: "The hairstyle has been successfully generated.");
        }

        printData("getImageId Data----------------$response");
        printData(
            "selectedImageId----------------${Get.find<HomeController>().selectedImageId}");

        update();
      } else {
        printData("Error-----------------");
      }
    });

    isLoading = false;
    update();
  }

  startRendering() async {
    isLoading = true;
    update();
    final eventId = getStorage.read("eventId");

    await HttpHandler.postHttpMethod(
        url: EndPoints().startRendering(eventId: eventId),
        data: {
          "hairstyle_id": Get.find<HomeController>().selectedImageId,
          "photo_ids": {"front": 0, "right": 1, "left": 2, "back": 3}
        }).then((value) {
      if (value["statusCode"] == 200) {
        final response = jsonDecode(value["body"]);

        printData("getImageId Data----------------$response");

        Get.find<TransFormationResultController>().getRenderingResult();
        update();
      } else {
        printData("Error-----------------");
      }
    });
    // Get.to(const TransFormationResultScreen());
    isLoading = false;
    update();
  }
}
