import 'dart:async';
import 'dart:convert';

import 'package:carousel_slider/carousel_slider.dart';
import 'package:get/get.dart';
import 'package:hair_style_app/confing/app_colors.dart';
import 'package:hair_style_app/confing/end_points.dart';
import 'package:hair_style_app/controller/face_photographing_controller.dart';
import 'package:hair_style_app/http_handler/http_handler.dart';
import 'package:hair_style_app/model/get_rendering_result.dart';
import 'package:hair_style_app/screens/transformation_result/transformation_result_screen.dart';

class TransFormationResultController extends GetxController {
  CarouselController buttonCarouselController = CarouselController();
  bool isLoading = false;
  bool isInProgress = false;
  GetRenderingResultModel? getRenderingResultModel;
  int tempIndex = 0;

  List<String> myImagesList = [];

  getRenderingResult() async {
    isLoading = true;
    update();
    final eventId = getStorage.read("eventId");
    printData("eventId:: $eventId");
    myImagesList = [];
    await HttpHandler.getHttpMethod(
      url: EndPoints().getRenderingResult(eventId: eventId),
    ).then((value) {
      print("value in rendering result:: $value");
      final response = jsonDecode(value["body"]);
      isInProgress = response["inprogress"];
      if (isInProgress) {
        Timer(const Duration(seconds: 2), () async {
          getRenderingResult();
        });
      } else {
        getRenderingResultModel =
            GetRenderingResultModel.fromJson(jsonDecode(value["body"]));
        printData(
            "Success-employeeData----------------$getRenderingResultModel");
        if (getRenderingResultModel?.hairstyleId?.front != "") {
          myImagesList.add(getRenderingResultModel!.hairstyleId!.front!);
        }

        // Add right image to the list
        if (getRenderingResultModel?.hairstyleId?.right != "") {
          myImagesList.add(getRenderingResultModel!.hairstyleId!.right!);
        }
        if (getRenderingResultModel?.hairstyleId?.left != "") {
          myImagesList.add(getRenderingResultModel!.hairstyleId!.left!);
        }
        if (getRenderingResultModel?.hairstyleId?.back != "") {
          myImagesList.add(getRenderingResultModel!.hairstyleId!.back!);
        }
        print("image :: $myImagesList");
        Get.to(const TransFormationResultScreen());
      }

      printData("getImageId Data----------------$response");
    });
    isLoading = false;
    update();
  }
}
