import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:hair_style_app/confing/app_colors.dart';
import 'package:hair_style_app/confing/constant.dart';
import 'package:hair_style_app/controller/face_photographing_controller.dart';
import 'package:hair_style_app/controller/upload_custom_image_controller.dart';
import 'package:qr_flutter/qr_flutter.dart';

class UploadCustomImageScreen extends StatelessWidget {
  const UploadCustomImageScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return GetBuilder(
      init: Get.find<UploadCustomImageController>(),
      builder: (controller) {
        return Scaffold(
          appBar: AppBar(
            automaticallyImplyLeading: false,
            surfaceTintColor: Colors.transparent,
            backgroundColor: AppColors.kf0f0f0fColor,
            title: Text(
              'upload_custom_image'.tr,
              style: const TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: AppColors.primaryColor),
            ),
            leading: GestureDetector(
                onTap: () {
                  Get.back();
                },
                child: Image.asset('assets/images/ic_back_button.png')),
          ),
          backgroundColor: AppColors.kf0f0f0fColor,
          body: controller.isLoading
              ? const Center(
                  child: CircularProgressIndicator(
                    color: AppColors.primaryColor,
                  ),
                )
              : Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 35),
                  child: Center(
                    child: Container(
                      height: size.height(235),
                      width: size.width(285),
                      decoration: BoxDecoration(
                          border: Border.all(
                              color: AppColors.k505050Color, width: 8)),
                      child: Center(
                        child: QrImageView(
                          data: controller.urlLink,
                          version: QrVersions.auto,
                          // size: 285,
                        ),
                      ),
                    ),
                  )),
          bottomNavigationBar: GestureDetector(
            onTap: () {
              Get.find<FacePhotoGraphingController>().getImageId();
            },
            child: Container(
              height: size.height(45),
              width: double.infinity,
              margin: const EdgeInsets.symmetric(horizontal: 28, vertical: 15),
              padding: const EdgeInsets.symmetric(vertical: 8),
              decoration: BoxDecoration(
                  color: AppColors.primaryColor,
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(color: AppColors.k030303Color, width: 4)),
              child: Center(
                  child: Text(
                'next_step'.tr,
                textAlign: TextAlign.center,
                style:
                    const TextStyle(fontSize: 24, color: AppColors.whiteColor),
              )),
            ),
          ),
        );
      },
    );
  }
}
