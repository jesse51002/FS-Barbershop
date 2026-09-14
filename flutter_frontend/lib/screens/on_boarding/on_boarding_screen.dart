import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:hair_style_app/confing/app_colors.dart';
import 'package:hair_style_app/confing/constant.dart';
import 'package:hair_style_app/controller/on_boarding_controller.dart';
import 'package:hair_style_app/screens/home/hair_style_selection.dart';

class OnBoardingScreen extends StatelessWidget {
  const OnBoardingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return GetBuilder(
      init: Get.find<OnBoardingController>(),
      builder: (controller) {
        return Scaffold(
          backgroundColor: AppColors.kf0f0f0fColor,
          body: controller.isLoading
              ? const Center(
                  child: CircularProgressIndicator(
                    color: AppColors.primaryColor,
                  ),
                )
              : SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      size.heightSpace(42),
                      Center(
                        child: Text("fusion_styles".tr,
                            style: const TextStyle(
                                fontSize: 40,
                                fontWeight: FontWeight.bold,
                                color: AppColors.k030303Color)),
                      ),
                      size.heightSpace(20),
                      const Divider(
                        color: AppColors.kDEDEDEColor,
                      ),
                      size.heightSpace(15),
                      Center(
                          child: Image.asset(
                        "assets/images/ic_logo.png",
                        height: size.height(167),
                        width: size.width(167),
                      )),
                      size.heightSpace(12),
                      Center(
                        child: Text("hairstyle_transformation_previewer".tr,
                            textAlign: TextAlign.center,
                            style: const TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color: AppColors.k030303Color)),
                      ),
                      size.heightSpace(28),
                      Padding(
                        padding: const EdgeInsets.only(left: 41, right: 41),
                        child: Center(
                          child: Text(
                              "tackle_the_unknowns_and_insecurities_of_changing_your_hair_style_using_our_tool_you_can_preview_how_you_ll_look_before_you_the_styling_is_done_so_you_can_proceed_with_confidence"
                                  .tr,
                              textAlign: TextAlign.center,
                              style: const TextStyle(
                                  fontSize: 16, color: AppColors.k030303Color)),
                        ),
                      ),
                      size.heightSpace(28),
                      GestureDetector(
                        onTap: () {
                          controller.getStarted();
                        },
                        child: Container(
                          width: size.width(254),
                          padding: const EdgeInsets.symmetric(vertical: 15),
                          decoration: BoxDecoration(
                              color: AppColors.primaryColor,
                              borderRadius: BorderRadius.circular(24),
                              border: Border.all(
                                  color: AppColors.k030303Color, width: 4)),
                          child: Center(
                              child: Text(
                            'create'.tr,
                            style: const TextStyle(
                                fontSize: 24, color: AppColors.whiteColor),
                          )),
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
