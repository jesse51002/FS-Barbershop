import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:hair_style_app/confing/app_colors.dart';
import 'package:hair_style_app/confing/constant.dart';
import 'package:hair_style_app/controller/splash_controller.dart';

class SplashScreen extends StatelessWidget {
  const SplashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return GetBuilder(
      init: Get.find<SplashController>(),
      builder: (controller) {
        return Scaffold(
          backgroundColor: AppColors.kf0f0f0fColor,
          body: GetBuilder(
            init: Get.find<SplashController>(),
            builder: (controller) {
              return Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Center(child: Image.asset("assets/images/ic_logo.png",height: size.height(200),width: size.width(200),)),
                  size.heightSpace(20),
                  Text("fusion_styles".tr, style: const TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.bold,
                    color: AppColors.k030303Color
                  ))
                ],
              );
            },
          ),
        );
      },
    );
  }
}
