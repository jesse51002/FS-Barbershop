class HairStylePreset {
  final List<Hairstyles>? hairstyles;

  HairStylePreset({
    this.hairstyles,
  });

  HairStylePreset.fromJson(Map<String, dynamic> json)
      : hairstyles = (json['Hairstyles'] as List?)
            ?.map((dynamic e) => Hairstyles.fromJson(e as Map<String, dynamic>))
            .toList();

  Map<String, dynamic> toJson() =>
      {'Hairstyles': hairstyles?.map((e) => e.toJson()).toList()};
}

class Hairstyles {
  final int? hairstlyeId;
  final String? hairstyleUrl;
  bool isSelected;

  Hairstyles({this.hairstlyeId, this.hairstyleUrl, this.isSelected = false});

  Hairstyles.fromJson(Map<String, dynamic> json)
      : hairstlyeId = json['hairstlye_id'] as int?,
        hairstyleUrl = json['hairstyle_url'] as String?,
        isSelected = json['isSelected'] ?? false;

  Map<String, dynamic> toJson() => {
        'hairstlye_id': hairstlyeId,
        'hairstyle_url': hairstyleUrl,
        "isSelected": isSelected
      };
}
