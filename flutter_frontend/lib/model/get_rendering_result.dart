class GetRenderingResultModel {
  final bool? inprogress;
  final HairstyleId? hairstyleId;

  GetRenderingResultModel({
    this.inprogress,
    this.hairstyleId,
  });

  GetRenderingResultModel.fromJson(Map<String, dynamic> json)
      : inprogress = json['inprogress'] as bool?,
        hairstyleId = (json['hairstyle_id'] as Map<String, dynamic>?) != null
            ? HairstyleId.fromJson(json['hairstyle_id'] as Map<String, dynamic>)
            : null;

  Map<String, dynamic> toJson() =>
      {'inprogress': inprogress, 'hairstyle_id': hairstyleId?.toJson()};
}

class HairstyleId {
  final String? front;
  final String? right;
  final String? left;
  final String? back;

  HairstyleId({
    this.front,
    this.right,
    this.left,
    this.back,
  });

  HairstyleId.fromJson(Map<String, dynamic> json)
      : front = json['front'] as String?,
        right = json['right'] as String?,
        left = json['left'] as String?,
        back = json['back'] as String?;

  Map<String, dynamic> toJson() =>
      {'front': front, 'right': right, 'left': left, 'back': back};
}
