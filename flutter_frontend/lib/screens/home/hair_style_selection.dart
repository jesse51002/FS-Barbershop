import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:get/get.dart';
import 'package:hair_style_app/confing/app_colors.dart';
import 'package:hair_style_app/confing/constant.dart';
import 'package:hair_style_app/controller/face_photographing_controller.dart';
import 'package:hair_style_app/controller/home_controller.dart';
import 'package:hair_style_app/controller/transformation_result_controller.dart';
import 'package:hair_style_app/controller/upload_custom_image_controller.dart';
import 'package:hair_style_app/screens/custom_image/upload_custom_image.dart';
import 'package:hair_style_app/screens/face_photographing/face_photographing_screen.dart';
import 'package:hair_style_app/screens/transformation_result/transformation_result_screen.dart';

class HairStyleSelectionScreen extends StatelessWidget {
  const HairStyleSelectionScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return GetBuilder(
      init: Get.find<HomeController>(),
      builder: (controller) {
        return Scaffold(
          appBar: AppBar(
            automaticallyImplyLeading: false,
            surfaceTintColor: Colors.transparent,
            backgroundColor: AppColors.kf0f0f0fColor,
            title: Text(
              'hair_style_selection'.tr,
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
          body: controller.isLoading ||
                  Get.find<TransFormationResultController>().isLoading
              ? const Center(
                  child: CircularProgressIndicator(
                    color: AppColors.primaryColor,
                  ),
                )
              : SingleChildScrollView(
                  child: Column(
                    children: [
                      const Divider(
                        color: AppColors.kDEDEDEColor,
                      ),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 28),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            size.heightSpace(25),
                            GestureDetector(
                              onTap: () {
                                Get.find<UploadCustomImageController>()
                                    .getQrUrl();
                              },
                              child: Container(
                                width: double.infinity,
                                padding:
                                    const EdgeInsets.symmetric(vertical: 8),
                                decoration: BoxDecoration(
                                    color: AppColors.kC7C7F1Color,
                                    borderRadius: BorderRadius.circular(24)),
                                child: Center(
                                    child: Text(
                                  'upload_custom_hairstyle'.tr,
                                  style: const TextStyle(
                                      color: AppColors.k030303Color,
                                      fontSize: 18),
                                )),
                              ),
                            ),
                            size.heightSpace(26),
                            Container(
                              height: size.height(425),
                              color: AppColors.kC2C2C2Color,
                              child: Scrollbar(
                                controller: controller.scrollController,
                                child: Center(
                                  child: GridView.builder(
                                    itemCount: controller
                                        .hairStylePreset?.hairstyles?.length,
                                    shrinkWrap: true,
                                    padding: const EdgeInsets.only(
                                        left: 9,
                                        top: 15,
                                        bottom: 15,
                                        right: 23),
                                    gridDelegate:
                                        const SliverGridDelegateWithFixedCrossAxisCount(
                                            crossAxisCount: 2,
                                            crossAxisSpacing: 26,
                                            mainAxisSpacing: 26),
                                    itemBuilder: (context, index) {
                                      return GestureDetector(
                                        onTap: () {
                                          controller.selectedImageIndex = index;

                                          for (var i = 0;
                                              i <
                                                  controller.hairStylePreset!
                                                      .hairstyles!.length;
                                              i++) {
                                            controller.hairStylePreset!
                                                    .hairstyles![i].isSelected =
                                                (i ==
                                                    controller
                                                        .selectedImageIndex);
                                          }

                                          controller.update();
                                          if (controller.selectedImageIndex !=
                                              -1) {
                                            controller.selectedImageId =
                                                controller
                                                    .hairStylePreset
                                                    ?.hairstyles?[index]
                                                    .hairstlyeId;
                                            controller.update();
                                            printData(
                                                "Selected Image ID: ${controller.selectedImageId}");
                                          }
                                        },
                                        child: Container(
                                          height: size.height(100),
                                          width: size.width(100),
                                          decoration: BoxDecoration(
                                              color: AppColors.kC7C7F1Color,
                                              borderRadius:
                                                  BorderRadius.circular(26),
                                              border: Border.all(
                                                  // color: controller
                                                  //             .selectedImageIndex ==
                                                  //         -1
                                                  color: controller
                                                              .selectedImageIndex ==
                                                          index
                                                      ? AppColors.primaryColor
                                                      : AppColors.k505050Color,
                                                  width: controller
                                                          .hairStylePreset!
                                                          .hairstyles![index]
                                                          .isSelected
                                                      ? 4
                                                      : 3)),
                                          child: ClipRRect(
                                            borderRadius:
                                                BorderRadius.circular(22),
                                            child: Image.network(
                                              controller
                                                      .hairStylePreset
                                                      ?.hairstyles?[index]
                                                      .hairstyleUrl ??
                                                  "",
                                              fit: BoxFit.cover,
                                            ),
                                          ),
                                        ),
                                      );
                                      // color: controller
                                    },
                                  ),
                                ),
                              ),
                            )
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
          bottomNavigationBar: GestureDetector(
            onTap: controller.selectedImageIndex == -1
                ? () {
                    // Fluttertoast.showToast(msg: "Please select hairstyle.");
                  }
                : () {
                    /// do needful operation here
                    printData("Tap on Button");
                    Get.find<TransFormationResultController>()
                            .myImagesList
                            .isNotEmpty
                        ? Get.find<FacePhotoGraphingController>()
                            .startRendering()
                        : null;

                    print(
                        "Tap Data --> ${Get.find<TransFormationResultController>().myImagesList.isNotEmpty}");

                    Get.find<TransFormationResultController>()
                            .myImagesList
                            .isNotEmpty
                        ? Get.to(const TransFormationResultScreen())
                        : Get.to(const FacePhotoGraphingScreen());
                  },
            child: Container(
              height: size.height(45),
              width: double.infinity,
              margin: const EdgeInsets.symmetric(horizontal: 28, vertical: 15),
              padding: const EdgeInsets.symmetric(vertical: 8),
              decoration: controller.selectedImageIndex == -1
                  ? BoxDecoration(
                      color: AppColors.kC7C7F1Color,
                      borderRadius: BorderRadius.circular(24))
                  : BoxDecoration(
                      color: AppColors.primaryColor,
                      borderRadius: BorderRadius.circular(24),
                      border:
                          Border.all(color: AppColors.k030303Color, width: 4)),
              child: Center(
                  child: Text(
                'select_hairstyle'.tr,
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 24, color: AppColors.whiteColor),
              )),
            ),
          ),
        );
      },
    );
  }
}
