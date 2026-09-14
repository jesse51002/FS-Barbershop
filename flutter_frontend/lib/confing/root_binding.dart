import 'package:get/get.dart';
import 'package:hair_style_app/controller/face_photographing_controller.dart';
import 'package:hair_style_app/controller/home_controller.dart';
import 'package:hair_style_app/controller/on_boarding_controller.dart';
import 'package:hair_style_app/controller/splash_controller.dart';
import 'package:hair_style_app/controller/transformation_result_controller.dart';
import 'package:hair_style_app/controller/upload_custom_image_controller.dart';

class RootBinding extends Bindings {
  @override
  void dependencies() {
    Get.put(SplashController());
    Get.put(HomeController());
    Get.put(OnBoardingController());
    Get.put(UploadCustomImageController());
    Get.put(FacePhotoGraphingController());
    Get.put(TransFormationResultController());
  }
}
