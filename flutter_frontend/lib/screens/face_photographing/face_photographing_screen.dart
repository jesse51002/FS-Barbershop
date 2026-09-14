import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:hair_style_app/confing/app_colors.dart';
import 'package:hair_style_app/confing/constant.dart';
import 'package:hair_style_app/controller/face_photographing_controller.dart';
import 'package:hair_style_app/controller/transformation_result_controller.dart';

class FacePhotoGraphingScreen extends StatelessWidget {
  const FacePhotoGraphingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return GetBuilder(
      init: Get.find<FacePhotoGraphingController>(),
      builder: (controller) {
        return Scaffold(
          appBar: AppBar(
            automaticallyImplyLeading: false,
            surfaceTintColor: Colors.transparent,
            backgroundColor: AppColors.kf0f0f0fColor,
            title: Text(
              'face_photographing'.tr,
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
                      size.heightSpace(10),
                      Center(
                          child: Text(
                        controller.tempIndex == 0
                            ? '"Front" Photo'
                            : controller.tempIndex == 1
                                ? '"Right" Photo'
                                : controller.tempIndex == 2
                                    ? '"Left" Photo'
                                    : controller.tempIndex == 3
                                        ? '"Back" Photo'
                                        : "",
                        style: const TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: AppColors.kC7C7F1Color),
                      )),
                      size.heightSpace(14),
                      Row(
                        children: [
                          Padding(
                            padding: const EdgeInsets.only(left: 0),
                            child: IconButton(
                              icon: const Icon(Icons.arrow_back_ios,
                                  color: AppColors.k505050Color, size: 30),
                              onPressed: () {
                                if (controller.tempIndex > 0) {
                                  controller.buttonCarouselController
                                      .previousPage();
                                  controller.tempIndex =
                                      controller.tempIndex - 1;
                                  controller.update();
                                }
                              },
                            ),
                          ),
                          Expanded(
                            child: Container(
                              height: size.height(288),
                              width: size.width(288),
                              decoration: BoxDecoration(
                                  border: Border.all(
                                      color: AppColors.k505050Color, width: 8)),
                              child: CarouselSlider.builder(
                                  itemCount: controller.myImagesList.length,
                                  carouselController:
                                      controller.buttonCarouselController,
                                  itemBuilder: (context, index, realIndex) {
                                    print(
                                        "5757--------${controller.myImagesList.length}");
                                    print("index--------${index}");
                                    print(
                                        "temp Index--------${controller.tempIndex}");
                                    return controller.myImagesList.isNotEmpty
                                        ? Image.file(
                                            controller.myImagesList[index],
                                            fit: BoxFit.cover,
                                          )
                                        : GestureDetector(
                                            onTap: () {
                                              controller.pickImage();
                                            },
                                            child: const Center(
                                              child: Column(
                                                mainAxisAlignment:
                                                    MainAxisAlignment.center,
                                                children: [
                                                  Icon(
                                                    Icons.camera_alt_outlined,
                                                    size: 40,
                                                    color:
                                                        AppColors.primaryColor,
                                                  ),
                                                  Text(
                                                    'Take "Front" Photo',
                                                    style: TextStyle(
                                                        fontWeight:
                                                            FontWeight.bold),
                                                  )
                                                ],
                                              ),
                                            ),
                                          );
                                  },
                                  disableGesture: true,
                                  options: CarouselOptions(
                                    height: size.height(288),
                                    enlargeCenterPage: true,
                                    autoPlay: false,
                                    viewportFraction: 1,
                                    aspectRatio: 1.0,
                                    reverse: false,
                                    scrollPhysics:
                                        const NeverScrollableScrollPhysics(),
                                    initialPage: 0,
                                    onPageChanged: (index, reason) {
                                      // controller.tempIndex = index;
                                      // controller.update();
                                      print("Carousel INDEX ===> $index");
                                    },
                                  )),
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.only(left: 0),
                            child: IconButton(
                              icon: const Icon(Icons.arrow_forward_ios,
                                  color: AppColors.k505050Color, size: 30),
                              onPressed: () {
                                if (controller.myImagesList.isNotEmpty) {
                                  print("tem${controller.tempIndex}");
                                  print(
                                      "myImagesListlength${controller.myImagesList.length}");
                                  if (controller.tempIndex <
                                      (controller.myImagesList.length - 1)) {
                                    controller.buttonCarouselController
                                        .nextPage();
                                    controller.tempIndex =
                                        controller.tempIndex + 1;
                                    controller.update();
                                  }
                                }
                              },
                            ),
                          ),
                        ],
                      ),
                      size.heightSpace(36),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 12),
                        child: Row(
                          children: [
                            Expanded(
                                child: GestureDetector(
                              onTap: () {
                                printData(
                                    "tempIndex :: ${controller.tempIndex}");
                                controller.myImagesList.isNotEmpty
                                    ? controller
                                        .replaceImage(controller.tempIndex)
                                    : null;
                              },
                              child: Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 10, vertical: 15),
                                decoration: controller.myImagesList.isNotEmpty
                                    ? BoxDecoration(
                                        color: AppColors.primaryColor,
                                        borderRadius: BorderRadius.circular(24),
                                        border: Border.all(
                                            color: AppColors.k030303Color,
                                            width: 4))
                                    : BoxDecoration(
                                        color: AppColors.kC7C7F1Color,
                                        borderRadius:
                                            BorderRadius.circular(24)),
                                child: Center(
                                    child: Text(
                                  controller.tempIndex == 0
                                      ? 'Edit "Front" Photo'
                                      : controller.tempIndex == 1
                                          ? 'Edit "Right" Photo'
                                          : controller.tempIndex == 2
                                              ? 'Edit "Left" Photo'
                                              : controller.tempIndex == 3
                                                  ? 'Edit "Back" Photo'
                                                  : "",
                                  style: const TextStyle(
                                      fontSize: 14,
                                      color: AppColors.whiteColor),
                                )),
                              ),
                            )),
                            size.widthSpace(10),
                            Expanded(
                                child: controller.tempIndex == 3
                                    ? const SizedBox()
                                    : GestureDetector(
                                        onTap: () {
                                          if (controller.myImagesList.length <=
                                              controller.tempIndex + 1) {
                                            if (controller
                                                    .myImagesList.length !=
                                                4) {
                                              controller.tempIndex + 1;
                                            } else {}
                                            if (controller
                                                    .myImagesList.length ==
                                                4) {
                                              // Get.to(const TransFormationResultScreen());
                                            } else {
                                              controller.isEnable
                                                  ? controller.myImagesList
                                                              .length <=
                                                          controller.tempIndex +
                                                              1
                                                      ? controller.pickImage()
                                                      : null
                                                  : null;
                                            }
                                          } else {
                                            printData("Not Possible");
                                          }
                                        },
                                        child: Container(
                                          padding: const EdgeInsets.symmetric(
                                              horizontal: 10, vertical: 15),
                                          decoration: controller.isEnable
                                              ? controller.myImagesList
                                                          .length <=
                                                      controller.tempIndex + 1
                                                  ? BoxDecoration(
                                                      color: AppColors
                                                          .primaryColor,
                                                      borderRadius:
                                                          BorderRadius.circular(
                                                              24),
                                                      border: Border.all(
                                                          color: AppColors
                                                              .k030303Color,
                                                          width: 4))
                                                  : BoxDecoration(
                                                      color: AppColors
                                                          .kC7C7F1Color,
                                                      borderRadius:
                                                          BorderRadius.circular(
                                                              24))
                                              : BoxDecoration(
                                                  color: AppColors.kC7C7F1Color,
                                                  borderRadius:
                                                      BorderRadius.circular(
                                                          24)),
                                          child: Center(
                                              child: Text(
                                            controller.tempIndex == 0
                                                ? 'Take "Right" Photo'
                                                : controller.tempIndex == 1
                                                    ? 'Take "Left" Photo'
                                                    : controller.tempIndex == 2
                                                        ? 'Take "Back" Photo'
                                                        : '',
                                            // 'Take "Next Dir" Photo',
                                            style: const TextStyle(
                                                fontSize: 14,
                                                color: AppColors.whiteColor),
                                          )),
                                        ),
                                      )),
                          ],
                        ),
                      ),
                      size.heightSpace(30),
                      GestureDetector(
                        onTap: () async {
                          if (controller.myImagesList.length == 4) {
                            controller.startRendering();
                          }
                        },
                        child: Container(
                          height: size.height(45),
                          width: double.infinity,
                          margin: const EdgeInsets.symmetric(horizontal: 28),
                          padding: const EdgeInsets.symmetric(vertical: 8),
                          decoration: controller.myImagesList.length != 4
                              ? BoxDecoration(
                                  color: AppColors.kC7C7F1Color,
                                  borderRadius: BorderRadius.circular(24))
                              : BoxDecoration(
                                  color: AppColors.primaryColor,
                                  borderRadius: BorderRadius.circular(24),
                                  border: Border.all(
                                      color: AppColors.k030303Color, width: 4)),
                          child: Center(
                              child: Text(
                            'finish_photographing'.tr,
                            textAlign: TextAlign.center,
                            style: const TextStyle(
                                fontSize: 24, color: AppColors.whiteColor),
                          )),
                        ),
                      ),
                      size.heightSpace(15),
                      const Padding(
                        padding: EdgeInsets.symmetric(horizontal: 20),
                        child: Text(
                          'Finish is grayed out until all directions have been captured. When finish is grayed out, Take "Next Dir" Photo should be made purple',
                          style: TextStyle(
                              fontSize: 14, color: AppColors.k030303Color),
                        ),
                      )
                    ],
                  ),
                ),
        );
      },
    );
  }
}
