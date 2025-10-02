// lib/shared/widgets/processing_banner.dart
import 'package:flutter/material.dart';

import '../services/pose_processing_controller.dart';

class ProcessingBanner extends StatelessWidget {
  const ProcessingBanner({
    super.key,
    required this.event,
    this.onCancel,
  });

  final ProgressEvent event;
  final VoidCallback? onCancel;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final textTheme = theme.textTheme;
    final radius = BorderRadius.circular(18);
    final isDeterminate = event.status == ProgressStatus.determinate;
    final isIndeterminate = event.status == ProgressStatus.indeterminate;
    final isError = event.status == ProgressStatus.error;
    final isComplete = event.status == ProgressStatus.complete;
    final phase = event.phase ?? (isError ? 'Error' : 'Processing');

    return Material(
      color: Colors.transparent,
      elevation: 8,
      borderRadius: radius,
      child: Container(
        width: 360,
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          borderRadius: radius,
          boxShadow: const [
            BoxShadow(
              color: Color(0x1A000000),
              blurRadius: 18,
              offset: Offset(0, 8),
            )
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  isError
                      ? Icons.error_outline
                      : isComplete
                          ? Icons.check_circle_outline
                          : Icons.blur_circular,
                  color: isError
                      ? theme.colorScheme.error
                      : theme.colorScheme.primary,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    phase,
                    style: textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            if (isIndeterminate)
              const LinearProgressIndicator(minHeight: 5)
            else if (isDeterminate)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  LinearProgressIndicator(
                    value: event.value?.clamp(0.0, 1.0),
                    minHeight: 5,
                  ),
                  const SizedBox(height: 6),
                  if (event.processed != null && event.total != null)
                    Text(
                      '${event.processed} of ${event.total} windows',
                      style: textTheme.bodySmall,
                    )
                  else
                    Text(
                      '${((event.value ?? 0) * 100).clamp(0, 100).toStringAsFixed(0)}%',
                      style: textTheme.bodySmall,
                    ),
                ],
              )
            else if (isError)
              Text(
                event.message ?? 'Processing failed.',
                style: textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.error,
                ),
              )
            else if (isComplete)
              Text(
                'All pose data is ready.',
                style: textTheme.bodyMedium,
              ),
            if (event.allowCancel &&
                onCancel != null &&
                (isDeterminate || isIndeterminate))
              Align(
                alignment: Alignment.centerRight,
                child: TextButton.icon(
                  onPressed: onCancel,
                  icon: const Icon(Icons.close_rounded),
                  label: const Text('Cancel'),
                  style: TextButton.styleFrom(
                    foregroundColor: theme.colorScheme.primary,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
