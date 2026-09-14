import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:hair_style_app/confing/app_colors.dart';
import 'package:hair_style_app/confing/end_points.dart';
import 'package:hair_style_app/controller/home_controller.dart';
import 'package:hair_style_app/http_handler/http_handler.dart';
import 'package:hair_style_app/screens/custom_image/upload_custom_image.dart';
import 'package:hair_style_app/screens/face_photographing/face_photographing_screen.dart';

class UploadCustomImageController extends GetxController {
  bool isLoading = false;
  String urlLink = "";
  bool isInProgress = false;

  getQrUrl() async {
    isLoading = true;
    update();
    final eventId = getStorage.read("eventId");
    urlLink = "";
    await HttpHandler.getHttpMethod(url: EndPoints().qrLink(eventId: eventId))
        .then((value) {
      if (value["statusCode"] == 200) {
        final response = jsonDecode(value["body"]);
        urlLink = response["link"];
        Get.to(const UploadCustomImageScreen());

        printData("getQrUrl Data----------------$response");
        printData("getQrUrl urlLink----------------$urlLink");

        update();
      } else {
        printData("Error-----------------");
      }
    });

    isLoading = false;
    update();
  }
}
