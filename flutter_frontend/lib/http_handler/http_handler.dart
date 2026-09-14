// ignore_for_file: unused_element
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:hair_style_app/confing/app_colors.dart';
import 'package:http/http.dart' as http;

class HttpHandler {
  static Map<String, String> getHeaders() {
    return {'Content-type': 'application/json', 'Accept': 'application/json'};
  }

  static Map<String, String> getHeaderWithToken() {
    return {
      'Content-type': 'application/json',
      'Accept': 'application/json',
    };
  }

  // Perform an HTTP GET request
  static Future<Map<String, dynamic>> getHttpMethod({required Uri url}) async {
    try {
      http.Response response = await http.get(
        url,
        headers: getHeaderWithToken(),
        // body: body == null ? null : jsonEncode(body),
      );
      printData("Get Response Code -- '${response.statusCode}'");
      printData("Get Response -- '${response.body}'");
      if (response.statusCode == 200) {
        printData("In Get '${response.statusCode}'");
        Map<String, dynamic> data = {
          'body': response.body,
          'headers': response.headers,
          'statusCode': response.statusCode,
          'error': null,
        };
        return data;
      } else {
        final dynamicWidth =
            (jsonDecode(response.body)['Result'].toString().length * 10)
                .toDouble();
        Get.rawSnackbar(
          titleText: const SizedBox(),
          maxWidth: dynamicWidth,
          snackStyle: SnackStyle.FLOATING,
          messageText: Center(
            child: Text(
              jsonDecode(response.body)['ResultText'] == null
                  ? jsonDecode(response.body)['Result']
                  : "${jsonDecode(response.body)['Result']} ${jsonDecode(response.body)['ResultText']}",
              style: const TextStyle(color: AppColors.whiteColor),
            ),
          ),
          margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
          padding:
              const EdgeInsets.only(bottom: 10, left: 10, right: 10, top: 3),
          borderRadius: 10,
          duration: const Duration(seconds: 3),
          backgroundColor: Colors.black.withOpacity(0.7),
          snackPosition: SnackPosition.BOTTOM,
        );
        printData("Error In Get '${response.body}'");
        printData("In Get 'else - ${response.statusCode}'");
        return {
          'body': response.body,
          'headers': response.headers,
          'error': "${response.statusCode}",
        };
      }
    } catch (e) {
      printData("In Get 'else - $e'");
      return {
        'body': '',
        'headers': '',
        'error': 404,
      };
    }
  }

  // Perform an HTTP POST request
  static Future<Map<String, dynamic>> postHttpMethod(
      {required Uri url, Map<String, dynamic>? data}) async {
    try {
      var header = getHeaderWithToken();
      printData("Post URL -- '$url'");
      printData("Post Data -- '$data'");
      printData("Post Header -- '$header'");
      http.Response response = await http.post(
        url,
        headers: header,
        body: data == null ? null : jsonEncode(data),
      );
      printData("POst API");
      printData("Post Response Code -- '${response.statusCode}'");

      printData("Post Response -- ${jsonDecode(response.body)}");
      if (response.statusCode == 200) {
        return {
          'body': response.body,
          'headers': response.headers,
          'statusCode': response.statusCode,
          'error': null,
        };
      } else {
        final dynamicWidth =
            (jsonDecode(response.body)['error']["message"].toString().length *
                    10)
                .toDouble();
        Get.rawSnackbar(
          titleText: const SizedBox(),
          maxWidth: dynamicWidth,
          snackStyle: SnackStyle.FLOATING,
          messageText: Center(
            child: Text(
              jsonDecode(response.body)['ResultText'] == null
                  ? jsonDecode(response.body)['error']["message"]
                      .toString()
                      .replaceAll("_", " ")
                  : "${jsonDecode(response.body)['Result']} ${jsonDecode(response.body)['ResultText']}",
              style: const TextStyle(color: AppColors.whiteColor),
            ),
          ),
          margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
          padding:
              const EdgeInsets.only(bottom: 10, left: 10, right: 10, top: 3),
          borderRadius: 10,
          duration: const Duration(seconds: 3),
          backgroundColor: Colors.black.withOpacity(0.7),
          snackPosition: SnackPosition.BOTTOM,
        );
        printData("In Post 'else - ${response.statusCode}'");
        return {
          'body': response.body,
          'headers': response.headers,
          'error': "${response.statusCode}",
        };
      }
    } catch (e) {
      return {
        'body': '$e',
        'headers': '',
        'error': 404,
      };
    }
  }

  // Perform an HTTP PATCH request
  static Future<Map<String, dynamic>> patchHttpMethod(
      {@required String? url, Map<String, dynamic>? data}) async {
    try {
      var header = getHeaders();
      printData("Patch URL -- '$url'");
      printData("Patch Data -- '$data'");
      printData("Patch Header -- '$header'");
      http.Response response = await http.patch(
        Uri.parse("url"),
        headers: header,
        body: data == null ? null : jsonEncode(data),
      );
      printData("Patch Response Code -- '${response.statusCode}'");
      printData("Patch Response -- '${response.body}'");
      if (response.statusCode == 200 || response.statusCode == 201) {
        printData("In Patch '${response.statusCode}'");
        Map<String, dynamic> data = {
          'body': response.body,
          'headers': response.headers,
          'error': null,
        };
        return data;
      } else {
        printData("In Patch 'else - ${response.statusCode}'");
        return {
          'body': response.body,
          'headers': response.headers,
          'error': "${response.statusCode}",
        };
      }
    } catch (e) {
      return {
        'body': '',
        'headers': '',
        'error': 404,
      };
    }
  }

  // Perform an HTTP PUT request
  static Future<Map<String, dynamic>> putHttpMethod(
      {@required String? url, Map<String, dynamic>? data}) async {
    try {
      var header = getHeaders();
      printData("Put URL -- '$url'");
      printData("Put Data -- '$data'");
      printData("Put Header -- '$header'");
      http.Response response = await http.put(
        Uri.parse("$url"),
        headers: header,
        body: data == null ? null : jsonEncode(data),
      );
      printData("PUT Response code -- '${response.statusCode}'");
      printData("PUT Response -- '${response.body}'");
      if (response.statusCode == 200 || response.statusCode == 201) {
        printData("In Put '${response.statusCode}'");
        Map<String, dynamic> data = {
          'body': response.body,
          'headers': response.headers,
          'error': null,
        };
        return data;
      } else {
        printData("In Put 'else - ${response.statusCode}'");
        return {
          'body': response.body,
          'headers': response.headers,
          'error': "${response.statusCode}",
        };
      }
    } catch (e) {
      return {
        'body': '',
        'headers': '',
        'error': 404,
      };
    }
  }

  // Perform an HTTP DELETE request
  static Future<Map<String, dynamic>> deleteHttpMethod(
      {@required String? url}) async {
    try {
      var header = getHeaders();
      printData("Delete URL -- '$url'");
      printData("Delete Data -- 'null'");
      printData("Delete Header -- '$header'");
      http.Response response = await http.delete(
        Uri.parse("$url"),
        headers: header,
      );
      printData("Delete Response Code -- '${response.statusCode}'");
      printData("Delete Response -- '${response.body}'");
      if (response.statusCode == 200 || response.statusCode == 201) {
        printData("In Delete '${response.statusCode}'");
        Map<String, dynamic> data = {
          'body': response.body,
          'headers': response.headers,
          'error': null,
        };
        return data;
      } else {
        printData("In Delete 'else - ${response.statusCode}'");
        return {
          'body': response.body,
          'headers': response.headers,
          'error': "${response.statusCode}",
        };
      }
    } catch (e) {
      return {
        'body': '',
        'headers': '',
        'error': 404,
      };
    }
  }

  // Perform an HTTP form data request
  static Future<Map<String, dynamic>> formHttpMethod({
    @required String? methodType,
    @required String? url,
    Map<String, String>? data,
    File? singleFile,
    String? singleFileKey,
  }) async {
    try {
      var header = getHeaders();
      printData("Form URL -- '$url'");
      printData("Form Header -- '$header'");
      http.MultipartRequest request =
          http.MultipartRequest(methodType!, Uri.parse("$url"));
      request.headers.addAll(header);
      if (data != null) {
        request.fields.addAll(data);
      }
      if (singleFile != null) {
        request.files.add(await http.MultipartFile.fromPath(
          singleFileKey!,
          singleFile.path,
        ));
      }
/*    if (singleFile2 != null) {
      request.files.add(await http.MultipartFile.fromPath(
        singleFileKey2!,
        singleFile2.path,
      ));
    }*/
      /*  if (multipleFile!.isNotEmpty) {
      for (File element in multipleFile) {
        request.files.add(await http.MultipartFile.fromPath(
          multipleFileKey!,
          element.path,
        ));
      }
    }*/

      printData("FORM FIELDS - ${request.fields}");
      printData("FORM FILES - ${request.files}");
      http.StreamedResponse streamedResponse = await request.send();
      if (streamedResponse.statusCode == 200 ||
          streamedResponse.statusCode == 201) {
        http.Response response =
            await http.Response.fromStream(streamedResponse);
        if (response.statusCode == 200 || response.statusCode == 201) {
          printData("In Post '${response.statusCode}'");
          printData("FORM RESPONSE -- '${response.body}'");
          // int status = json.decode(response.body)["status"];
          if (kDebugMode) {
            printData("STATUS _+_+_+_+_+_+_ ${response.statusCode}");
          }
          Map<String, dynamic> data;
          if (response.statusCode == 1) {
            data = {
              'body': response.body,
              'headers': response.headers,
              'error': null,
            };
          } else {
            data = {
              'body': response.body,
              'headers': response.headers,
              'error': response.statusCode,
            };
          }
          return data;
        } else {
          printData("In Form 'else - ${response.statusCode}'");
          return {
            'body': response.body,
            'headers': response.headers,
            'error': "${response.statusCode}",
          };
        }
      } else {
        printData("In Form 'else ---- ${streamedResponse.statusCode}'");
        return {
          'headers': streamedResponse.headers,
          'error': "${streamedResponse.statusCode}",
        };
      }
    } catch (e) {
      return {
        'body': '',
        'headers': '',
        'error': 404,
      };
    }
  }

/*  upload(File imageFile) async {
    // open a bytestream
    var stream = new http.ByteStream(DelegatingStream.typed(imageFile.openRead()));
    // get file length
    var length = await imageFile.length();

    // string to uri
    var uri = Uri.parse("http://ip:8082/composer/predict");

    // create multipart request
    var request = new http.MultipartRequest("POST", uri);

    // multipart that takes file
    var multipartFile = http.MultipartFile('file', stream, length,
        filename: imageFile.path
        // filename: basename(imageFile.path)

    );

    // add file to multipart
    request.files.add(multipartFile);

    // send
    var response = await request.send();
    print(response.statusCode);

    // listen for response
    response.stream.transform(utf8.decoder).listen((value) {
      print(value);
    });
  }*/
}
