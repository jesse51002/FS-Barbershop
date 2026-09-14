import 'dart:convert';

import 'package:get/get.dart';
import 'package:hair_style_app/confing/app_colors.dart';
import 'package:hair_style_app/confing/end_points.dart';
import 'package:hair_style_app/controller/home_controller.dart';
import 'package:hair_style_app/http_handler/http_handler.dart';
import 'package:hair_style_app/screens/home/hair_style_selection.dart';

class OnBoardingController extends GetxController {
  bool isLoading = false;

  getStarted() async {
    isLoading = true;
    update();

    await HttpHandler.getHttpMethod(url: EndPoints().getStarted())
        .then((value) {
      if (value["statusCode"] == 200) {
        final response = jsonDecode(value["body"]);
        getStorage.write("eventId", response["eventid"]);
        printData("getStarted Data----------------$response");
        Get.to(const HairStyleSelectionScreen());
        update();
        Get.find<HomeController>().hairStylePresetData();
      } else {
        printData("Error----------------- Get Started");
      }
    });

    isLoading = false;
    update();
  }
}
