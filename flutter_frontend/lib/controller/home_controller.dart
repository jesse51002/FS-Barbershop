import 'dart:convert';

import 'package:flutter/cupertino.dart';
import 'package:get/get.dart';
import 'package:hair_style_app/confing/app_colors.dart';
import 'package:hair_style_app/confing/end_points.dart';
import 'package:hair_style_app/http_handler/http_handler.dart';
import 'package:hair_style_app/model/hair_style_preset.dart';

class HomeController extends GetxController {
  final ScrollController scrollController = ScrollController();
  HairStylePreset? hairStylePreset;
  bool isLoading = false;
  int selectedImageIndex = -1;
  int? selectedImageId;
  hairStylePresetData() async {
    isLoading = true;
    selectedImageIndex = -1;
    update();
    final eventId = getStorage.read("eventId");
    await HttpHandler.getHttpMethod(
            url: EndPoints().hairStylePreset(eventId: eventId))
        .then((value) {
      if (value["statusCode"] == 200) {
        hairStylePreset = HairStylePreset.fromJson(jsonDecode(value["body"]));
        printData("Success-employeeData----------------$hairStylePreset");

        update();
      } else {
        printData("Error-----------------");
      }
    });
    isLoading = false;
    update();
  }
}
