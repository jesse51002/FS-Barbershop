import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:hair_style_app/confing/app_colors.dart';
import 'package:hair_style_app/confing/constant.dart';
import 'package:hair_style_app/controller/face_photographing_controller.dart';
import 'package:hair_style_app/controller/home_controller.dart';
import 'package:hair_style_app/controller/transformation_result_controller.dart';
import 'package:hair_style_app/screens/home/hair_style_selection.dart';
import 'package:hair_style_app/screens/on_boarding/on_boarding_screen.dart';

class TransFormationResultScreen extends StatelessWidget {
  const TransFormationResultScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return GetBuilder(
      init: Get.find<TransFormationResultController>(),
      builder: (controller) {
        return Scaffold(
          appBar: AppBar(
            automaticallyImplyLeading: false,
            surfaceTintColor: Colors.transparent,
            backgroundColor: AppColors.kf0f0f0fColor,
            centerTitle: true,
            title: Text(
              'transformation_results'.tr,
              style: const TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: AppColors.primaryColor),
            ),
          ),
          backgroundColor: AppColors.kf0f0f0fColor,
          body: controller.isLoading
              ? const Center(
                  child: CircularProgressIndicator(
                    color: AppColors.primaryColor,
                  ),
                )
              : Column(
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
                          padding: const EdgeInsets.only(left: 8),
                          child: IconButton(
                            icon: const Icon(Icons.arrow_back_ios,
                                color: AppColors.k505050Color, size: 30),
                            onPressed: () {
                              if (controller.tempIndex != 0) {
                                controller.buttonCarouselController
                                    .previousPage();
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
                            child: controller.myImagesList.isEmpty
                                ? const Center(
                                    child: CircularProgressIndicator(),
                                  )
                                : CarouselSlider.builder(
                                    itemCount: controller.myImagesList.length,
                                    carouselController:
                                        controller.buttonCarouselController,
                                    itemBuilder: (context, index, realIndex) {
                                      return Image.network(
                                        controller.myImagesList[index],
                                        fit: BoxFit.cover,
                                      );
                                    },
                                    disableGesture: true,
                                    options: CarouselOptions(
                                      height: size.height(288),
                                      enlargeCenterPage: true,
                                      autoPlay: false,
                                      viewportFraction: 1,
                                      aspectRatio: 1.0,
                                      scrollPhysics:
                                          const NeverScrollableScrollPhysics(),
                                      initialPage: 0,
                                      onPageChanged: (index, reason) {
                                        controller.tempIndex = index;
                                        controller.update();
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
                              if (controller.tempIndex != 3) {
                                controller.buttonCarouselController.nextPage();
                                controller.update();
                              }
                            },
                          ),
                        ),
                      ],
                    ),
                    size.heightSpace(55),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      child: Row(
                        children: [
                          Expanded(
                              child: GestureDetector(
                            onTap: () {
                              Get.find<FacePhotoGraphingController>().update();
                              Get.to(const HairStyleSelectionScreen());
                            },
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 10, vertical: 15),
                              decoration: BoxDecoration(
                                  color: AppColors.primaryColor,
                                  borderRadius: BorderRadius.circular(24),
                                  border: Border.all(
                                      color: AppColors.k030303Color, width: 4)),
                              child: Center(
                                  child: Text(
                                'try_another_style'.tr,
                                style: const TextStyle(
                                    fontSize: 12, color: AppColors.whiteColor),
                              )),
                            ),
                          )),
                          size.widthSpace(10),
                          Expanded(
                              child: GestureDetector(
                            onTap: () {
                              Get.find<FacePhotoGraphingController>()
                                  .tempIndex = 0;
                              Get.find<FacePhotoGraphingController>()
                                  .myImagesList
                                  .clear();
                              Get.find<FacePhotoGraphingController>().isEnable =
                                  false;
                              Get.find<TransFormationResultController>()
                                  .myImagesList
                                  .clear();
                              Get.find<HomeController>().selectedImageIndex =
                                  -1;
                              Get.find<HomeController>().update();
                              Get.find<FacePhotoGraphingController>().update();
                              Get.find<TransFormationResultController>()
                                  .update();
                              Get.offAll(const OnBoardingScreen());
                            },
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 10, vertical: 15),
                              decoration: BoxDecoration(
                                  color: AppColors.primaryColor,
                                  borderRadius: BorderRadius.circular(24),
                                  border: Border.all(
                                      color: AppColors.k030303Color, width: 4)),
                              child: Center(
                                  child: Text(
                                'finish'.tr,
                                style: const TextStyle(
                                    fontSize: 12, color: AppColors.whiteColor),
                              )),
                            ),
                          )),
                        ],
                      ),
                    ),
                    size.heightSpace(36),
                    const Padding(
                      padding: EdgeInsets.symmetric(horizontal: 20),
                      child: Text(
                        'Try another style saves Face Photographs and skips the step',
                        style: TextStyle(
                            fontSize: 14, color: AppColors.k030303Color),
                      ),
                    )
                  ],
                ),
        );
      },
    );
  }
}
