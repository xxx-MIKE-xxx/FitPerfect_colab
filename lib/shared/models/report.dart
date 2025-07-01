// A dead-simple DTO for the analysis JSON.
//
// #2  Serializer helper (tutorial)
class Report {
  Report({
    required this.fps,
    required this.mistakes,
    required this.raw,
  });

  final double fps;
  final List<Mistake> mistakes;

  /// Keep the original map around in case you need something
  /// the typed model doesn’t expose yet.
  final Map<String, dynamic> raw;

  factory Report.fromJson(Map<String, dynamic> json) => Report(
        fps      : (json['fps'] as num?)?.toDouble() ?? 30,
        mistakes : (json['mistakes'] as List<dynamic>? ?? [])
            .map((m) => Mistake.fromJson(m as Map<String, dynamic>))
            .toList(),
        raw      : json,
      );
}

class Mistake {
  Mistake({
    required this.frame,
    required this.message,
  });

  final int    frame;
  final String message;

  factory Mistake.fromJson(Map<String, dynamic> json) => Mistake(
        frame   : json['frame']   as int,
        message : json['message'] as String,
      );
}
