// lib/screens/progress_screen.dart
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:go_router/go_router.dart';
import 'package:share_plus/share_plus.dart';

/// ProgressScreen – 3-module layout:
/// - Overview (default)
/// - Calendar
/// - History
///
/// Notes:
/// • No extra packages required. Calendar is a custom month grid w/ heat-map.
/// • All data is mocked; replace the *_DAO calls with your repos later.
/// • Analytics hooks: debugPrint events for tab change, day taps, filters.

class ProgressScreen extends StatefulWidget {
  const ProgressScreen({super.key});

  @override
  State<ProgressScreen> createState() => _ProgressScreenState();
}

class _ProgressScreenState extends State<ProgressScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tab;
  final _historyKey = GlobalKey<_HistoryListState>();

  // ---- Mock data -------------------------------------------------------------

  // Pretend sessions for the last ~60 days
  late final List<_Session> _sessions = _MockData.generateSessions();

  // Derived mock metrics
  int get _avgScore7d {
    final now = DateTime.now();
    final week = _sessions
        .where((s) => s.date.isAfter(now.subtract(const Duration(days: 7))));
    if (week.isEmpty) return 0;
    return (week.map((s) => s.score).reduce((a, b) => a + b) / week.length)
        .round()
        .clamp(0, 100);
  }

  List<int> get _weeklyGoalStreak {
    // last 10 ISO weeks: mark 1 if >=2 sessions that week, else 0
    final now = DateTime.now();
    final res = <int>[];
    for (int i = 9; i >= 0; i--) {
      final start = now.subtract(Duration(days: now.weekday - 1 + i * 7));
      final end = start.add(const Duration(days: 7));
      final cnt = _sessions.where((s) => s.date.isAfter(start) && s.date.isBefore(end)).length;
      res.add(cnt >= 2 ? 1 : 0);
    }
    return res;
  }

  _Badge get _latestBadge => _MockData.latestBadge;
  List<_Badge> get _allBadges => _MockData.allBadges;
  List<_CoachTip> get _coachTips => _MockData.coachTips;

  @override
  void initState() {
    super.initState();
    _tab = TabController(length: 3, vsync: this);
    _tab.addListener(() {
      if (_tab.indexIsChanging) return;
      debugPrint('analytics: progress_tab_view tab=${_tab.index}');
    });
  }

  @override
  void dispose() {
    _tab.dispose();
    super.dispose();
  }

  void _shareScore() {
    Share.share('My form score this week: $_avgScore7d/100 💪');
  }

  // When calendar long-presses a day, jump the history list to that date.
  void _jumpHistoryTo(DateTime day) {
    _tab.animateTo(2);
    // Defer to next frame so history is built
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _historyKey.currentState?.scrollToDate(day);
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final titleStyle = theme.textTheme.titleLarge?.copyWith(
      fontWeight: FontWeight.w800,
      letterSpacing: -.2,
    );

    return Scaffold(
      appBar: AppBar(
        title: Text('Progress', style: titleStyle),
        centerTitle: false,
        bottom: TabBar(
          controller: _tab,
          isScrollable: true,
          labelPadding: const EdgeInsets.symmetric(horizontal: 14),
          indicatorSize: TabBarIndicatorSize.label,
          indicator: const UnderlineTabIndicator(
            borderSide: BorderSide(width: 4, color: Color(0xFF00C6A2)),
          ),
          labelStyle: const TextStyle(fontWeight: FontWeight.w800),
          tabs: const [
            Tab(text: 'Overview'),
            Tab(text: 'Calendar'),
            Tab(text: 'History'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tab,
        children: [
          // ---------------- OVERVIEW ----------------
          _OverviewTab(
            avgScore7d: _avgScore7d,
            weeklyStreak: _weeklyGoalStreak,
            latestBadge: _latestBadge,
            allBadges: _allBadges,
            tips: _coachTips,
            latestReelThumb: _MockData.reelThumb,
            onShare: _shareScore,
          ),

          // ---------------- CALENDAR ----------------
          _CalendarTab(
            sessions: _sessions,
            onDayTap: (day, sessions) {
              debugPrint('analytics: calendar_day_tap date=$day has=${sessions.isNotEmpty}');
              _CalendarBottomSheet.show(context, day, sessions);
            },
            onDayLongPress: _jumpHistoryTo,
          ),

          // ---------------- HISTORY ----------------
          _HistoryList(
            key: _historyKey,
            sessions: _sessions,
          ),
        ],
      ),
    );
  }
}

/* ============================================================================
 * OVERVIEW TAB
 * ==========================================================================*/

class _OverviewTab extends StatelessWidget {
  const _OverviewTab({
    required this.avgScore7d,
    required this.weeklyStreak,
    required this.latestBadge,
    required this.allBadges,
    required this.tips,
    required this.latestReelThumb,
    required this.onShare,
  });

  final int avgScore7d;
  final List<int> weeklyStreak;
  final _Badge latestBadge;
  final List<_Badge> allBadges;
  final List<_CoachTip> tips;
  final String latestReelThumb;
  final VoidCallback onShare;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _SectionTitle('Overall score'),
          const SizedBox(height: 8),
          _ScoreHeroCard(score: avgScore7d, onShare: onShare),

          const SizedBox(height: 16),
          _SectionTitle('Weekly goal streak'),
          const SizedBox(height: 8),
          _StreakStrip(weeks: weeklyStreak),

          const SizedBox(height: 16),
          _SectionTitle('Latest badge'),
          const SizedBox(height: 8),
          _LatestBadgeTile(badge: latestBadge),

          const SizedBox(height: 16),
          _SectionTitle('Coach suggestions'),
          const SizedBox(height: 8),
          _SuggestionsCarousel(tips: tips),

          const SizedBox(height: 16),
          _SectionTitle('Highlight reel'),
          const SizedBox(height: 8),
          _ReelThumb(src: latestReelThumb),

          const SizedBox(height: 16),
          _SectionTitle('Badges to unlock'),
          const SizedBox(height: 8),
          _BadgesGrid(badges: allBadges),
        ],
      ),
    );
  }
}

class _SectionTitle extends StatelessWidget {
  const _SectionTitle(this.text);
  final String text;
  @override
  Widget build(BuildContext context) => Text(
        text,
        style: Theme.of(context).textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w800,
            ),
      );
}

class _ScoreHeroCard extends StatelessWidget {
  const _ScoreHeroCard({required this.score, required this.onShare});
  final int score;
  final VoidCallback onShare;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final pct = (score / 100).clamp(0.0, 1.0);

    return Material(
      elevation: 6,
      borderRadius: BorderRadius.circular(20),
      color: theme.colorScheme.surface,
      child: Container(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            SizedBox(
              width: 88,
              height: 88,
              child: CustomPaint(
                painter: _RingPainter(pct),
                child: Center(
                  child: Text(
                    '$score',
                    style: theme.textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.w900,
                      fontFeatures: const [FontFeature.tabularFigures()],
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: DefaultTextStyle(
                style: theme.textTheme.bodyMedium!,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Excellent form', style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800)),
                    const SizedBox(height: 4),
                    Text('Minor tweaks only', style: theme.textTheme.bodyMedium?.copyWith(color: theme.hintColor)),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Icon(Icons.trending_up, color: Colors.green.shade600, size: 18),
                        const SizedBox(width: 6),
                        Text('7-day avg', style: theme.textTheme.labelMedium),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            IconButton(
              tooltip: 'Share',
              onPressed: onShare,
              icon: const Icon(Icons.ios_share_rounded),
            ),
          ],
        ),
      ),
    );
  }
}

class _RingPainter extends CustomPainter {
  _RingPainter(this.pct);
  final double pct;

  @override
  void paint(Canvas canvas, Size size) {
    final stroke = 10.0;
    final rect = Offset.zero & size;
    final center = rect.center;
    final radius = math.min(size.width, size.height) / 2 - stroke / 2;

    // background track
    final track = Paint()
      ..color = const Color(0x11000000)
      ..style = PaintingStyle.stroke
      ..strokeWidth = stroke;
    canvas.drawCircle(center, radius, track);

    // gradient arc
    final sweep = 2 * math.pi * pct;
    final shader = SweepGradient(
      startAngle: -math.pi / 2,
      endAngle: -math.pi / 2 + sweep,
      colors: const [Color(0xFFFFC107), Color(0xFFFF7043), Color(0xFFE91E63)],
    ).createShader(Rect.fromCircle(center: center, radius: radius));
    final fg = Paint()
      ..shader = shader
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeWidth = stroke;
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      -math.pi / 2,
      sweep,
      false,
      fg,
    );
  }

  @override
  bool shouldRepaint(covariant _RingPainter oldDelegate) =>
      oldDelegate.pct != pct;
}

class _StreakStrip extends StatelessWidget {
  const _StreakStrip({required this.weeks});
  final List<int> weeks;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 44,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: weeks.length,
        separatorBuilder: (_, __) => const SizedBox(width: 10),
        itemBuilder: (ctx, i) {
          final met = weeks[i] == 1;
          return AnimatedContainer(
            duration: const Duration(milliseconds: 300),
            padding: const EdgeInsets.symmetric(horizontal: 12),
            decoration: BoxDecoration(
              color: met ? const Color(0xFF00C853) : const Color(0xFFE0E0E0),
              borderRadius: BorderRadius.circular(999),
              boxShadow: met
                  ? [const BoxShadow(color: Color(0x3300C853), blurRadius: 12, offset: Offset(0, 4))]
                  : null,
            ),
            alignment: Alignment.center,
            child: Text(
              'W${i + 1}',
              style: TextStyle(
                color: met ? Colors.white : Colors.black87,
                fontWeight: FontWeight.w700,
              ),
            ),
          );
        },
      ),
    );
  }
}

class _LatestBadgeTile extends StatelessWidget {
  const _LatestBadgeTile({required this.badge});
  final _Badge badge;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Material(
      color: theme.colorScheme.surface,
      elevation: 3,
      borderRadius: BorderRadius.circular(16),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: badge.bg,
          child: Icon(badge.icon, color: Colors.white),
        ),
        title: Text(badge.title, style: const TextStyle(fontWeight: FontWeight.w800)),
        subtitle: Text('Unlocked ${DateFormat.MMMd().format(badge.date)}'),
        trailing: const Icon(Icons.chevron_right),
        onTap: () {},
      ),
    );
  }
}

class _SuggestionsCarousel extends StatelessWidget {
  const _SuggestionsCarousel({required this.tips});
  final List<_CoachTip> tips;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 108,
      child: PageView.builder(
        controller: PageController(viewportFraction: .9),
        itemCount: math.min(3, tips.length),
        itemBuilder: (_, i) {
          final t = tips[i];
          return Padding(
            padding: const EdgeInsets.only(right: 8),
            child: Material(
              elevation: 3,
              borderRadius: BorderRadius.circular(16),
              color: Colors.white,
              child: ListTile(
                leading: CircleAvatar(
                  backgroundColor: t.color.withOpacity(.15),
                  child: Icon(t.icon, color: t.color),
                ),
                title: Text(t.title, style: const TextStyle(fontWeight: FontWeight.w800)),
                subtitle: Text(t.subtitle),
                trailing: const Icon(Icons.chevron_right),
                onTap: () {},
              ),
            ),
          );
        },
      ),
    );
  }
}

class _ReelThumb extends StatelessWidget {
  const _ReelThumb({required this.src});
  final String src;
  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: 16 / 9,
      child: Stack(
        children: [
          // Placeholder reel thumbnail (solid gradient)
          Container(
            decoration: const BoxDecoration(
              borderRadius: BorderRadius.all(Radius.circular(16)),
              gradient: LinearGradient(
                colors: [Color(0xFF2196F3), Color(0xFF673AB7)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
            ),
          ),
          const Positioned.fill(
            child: Center(
              child: Icon(Icons.play_circle_fill, size: 72, color: Colors.white),
            ),
          ),
        ],
      ),
    );
  }
}

class _BadgesGrid extends StatelessWidget {
  const _BadgesGrid({required this.badges});
  final List<_Badge> badges;

  @override
  Widget build(BuildContext context) {
    return GridView.builder(
      physics: const NeverScrollableScrollPhysics(),
      shrinkWrap: true,
      itemCount: badges.length,
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 4,
        mainAxisSpacing: 10,
        crossAxisSpacing: 10,
      ),
      itemBuilder: (_, i) {
        final b = badges[i];
        return Column(
          children: [
            Container(
              width: 56,
              height: 56,
              decoration: BoxDecoration(
                color: b.bg,
                shape: BoxShape.circle,
                boxShadow: const [BoxShadow(color: Color(0x22000000), blurRadius: 8)],
              ),
              child: Icon(b.icon, color: Colors.white),
            ),
            const SizedBox(height: 6),
            Text(
              b.short,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600),
            ),
          ],
        );
      },
    );
  }
}

/* ============================================================================
 * CALENDAR TAB
 * ==========================================================================*/

class _CalendarTab extends StatefulWidget {
  const _CalendarTab({
    required this.sessions,
    required this.onDayTap,
    required this.onDayLongPress,
  });

  final List<_Session> sessions;
  final void Function(DateTime day, List<_Session> sessions) onDayTap;
  final void Function(DateTime day) onDayLongPress;

  @override
  State<_CalendarTab> createState() => _CalendarTabState();
}

class _CalendarTabState extends State<_CalendarTab> {
  late DateTime _focusedMonth;

  @override
  void initState() {
    super.initState();
    _focusedMonth = DateTime(DateTime.now().year, DateTime.now().month);
  }

  void _prev() {
    setState(() {
      _focusedMonth = DateTime(_focusedMonth.year, _focusedMonth.month - 1);
    });
  }

  void _next() {
    setState(() {
      _focusedMonth = DateTime(_focusedMonth.year, _focusedMonth.month + 1);
    });
  }

  @override
  Widget build(BuildContext context) {
    final firstDayOfMonth = DateTime(_focusedMonth.year, _focusedMonth.month, 1);
    final daysInMonth = DateUtils.getDaysInMonth(_focusedMonth.year, _focusedMonth.month);
    final startWeekday = firstDayOfMonth.weekday; // 1..7 (Mon..Sun)
    final cells = <DateTime?>[];

    // Pad from Monday
    for (int i = 1; i < startWeekday; i++) {
      cells.add(null);
    }
    for (int d = 1; d <= daysInMonth; d++) {
      cells.add(DateTime(_focusedMonth.year, _focusedMonth.month, d));
    }

    // Map date → sessions
    Map<String, List<_Session>> byDay = {};
    for (final s in widget.sessions) {
      final key = DateFormat('yyyy-MM-dd').format(s.date);
      (byDay[key] ??= []).add(s);
    }

    return Column(
      children: [
        // Month header
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
          child: Row(
            children: [
              IconButton(onPressed: _prev, icon: const Icon(Icons.chevron_left)),
              Expanded(
                child: Center(
                  child: Text(
                    DateFormat.yMMMM().format(_focusedMonth),
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800),
                  ),
                ),
              ),
              IconButton(onPressed: _next, icon: const Icon(Icons.chevron_right)),
            ],
          ),
        ),

        // Weekday labels
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Row(
            children: List.generate(7, (i) {
              final label = DateFormat.E().format(DateTime(2024, 1, i + 1));
              return Expanded(
                child: Center(
                  child: Text(label.substring(0, 2), style: const TextStyle(fontWeight: FontWeight.w700)),
                ),
              );
            }),
          ),
        ),
        const SizedBox(height: 8),

        // Grid
        Expanded(
          child: GridView.builder(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 7,
              crossAxisSpacing: 8,
              mainAxisSpacing: 8,
            ),
            itemCount: cells.length,
            itemBuilder: (_, i) {
              final day = cells[i];
              if (day == null) {
                return const SizedBox.shrink();
              }
              final key = DateFormat('yyyy-MM-dd').format(day);
              final list = byDay[key] ?? const <_Session>[];

              final intensity = _heatValue(list);
              final bg = _heatColor(intensity);

              return GestureDetector(
                onTap: () => widget.onDayTap(day, list),
                onLongPress: () => widget.onDayLongPress(day),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 180),
                  decoration: BoxDecoration(
                    color: bg,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: Colors.black12),
                  ),
                  child: Center(
                    child: Text(
                      '${day.day}',
                      style: TextStyle(
                        fontWeight: FontWeight.w700,
                        color: intensity > 0.4 ? Colors.white : Colors.black87,
                        fontFeatures: const [FontFeature.tabularFigures()],
                      ),
                    ),
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  // Combine frequency + score into a 0..1 value
  double _heatValue(List<_Session> list) {
    if (list.isEmpty) return 0;
    final avgScore = list.map((e) => e.score).reduce((a, b) => a + b) / list.length;
    final freq = list.length.clamp(0, 3) / 3.0;
    return (avgScore / 100) * 0.6 + freq * 0.4;
  }

  Color _heatColor(double v) {
    if (v == 0) return const Color(0xFFF3F4F6);
    // Gradient grey → teal → green
    return Color.lerp(const Color(0xFF80CBC4), const Color(0xFF00C853), v)!;
  }
}

class _CalendarBottomSheet {
  static void show(BuildContext context, DateTime day, List<_Session> list) {
    showModalBottomSheet(
      context: context,
      showDragHandle: true,
      builder: (_) {
        if (list.isEmpty) {
          return Padding(
            padding: const EdgeInsets.all(16),
            child: Text('No sessions on ${DateFormat.yMMMd().format(day)}'),
          );
        }
        return ListView.separated(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          itemCount: list.length,
          separatorBuilder: (_, __) => const SizedBox(height: 10),
          itemBuilder: (_, i) {
            final s = list[i];
            return ListTile(
              leading: _Thumb(score: s.score),
              title: Text('${s.exercise} · ${DateFormat.Hm().format(s.date)}',
                  style: const TextStyle(fontWeight: FontWeight.w700)),
              subtitle: Text('Score ${s.score} · ${s.reps} reps · ${s.weight} kg'),
              trailing: const Icon(Icons.open_in_new),
              onTap: () {
                // Deep link to Feedback (mock extras)
                context.go('/feedback', extra: {
                  'videoPath': null,
                  'videoKey': s.s3Key ?? 'local/mock/${s.id}.mp4',
                  'report': const {'data': []},
                });
              },
            );
          },
        );
      },
    );
  }
}

/* ============================================================================
 * HISTORY TAB
 * ==========================================================================*/

class _HistoryList extends StatefulWidget {
  const _HistoryList({super.key, required this.sessions});
  final List<_Session> sessions;

  @override
  State<_HistoryList> createState() => _HistoryListState();
}

class _HistoryListState extends State<_HistoryList> {
  final ScrollController _scroll = ScrollController();

  // Group sessions by month
  late final Map<String, List<_Session>> _byMonth = () {
    final map = <String, List<_Session>>{};
    for (final s in widget.sessions) {
      final key = DateFormat('yyyy-MM').format(s.date);
      (map[key] ??= []).add(s);
    }
    // sort inside groups (newest first)
    for (final k in map.keys) {
      map[k]!.sort((a, b) => b.date.compareTo(a.date));
    }
    // sort keys (newest first)
    final sorted = Map.fromEntries(
      map.entries.toList()
        ..sort((a, b) => b.key.compareTo(a.key)),
    );
    return sorted;
  }();

  // Jump to the first item of a date's month
  void scrollToDate(DateTime date) {
    final targetKey = DateFormat('yyyy-MM').format(date);
    double offset = 0;
    for (final entry in _byMonth.entries) {
      // header + items
      offset += 48; // header height
      if (entry.key == targetKey) break;
      offset += entry.value.length * 90.0 + entry.value.length * 8.0; // tile + spacing
    }
    _scroll.animateTo(
      offset,
      duration: const Duration(milliseconds: 350),
      curve: Curves.easeOutCubic,
    );
  }

  @override
  Widget build(BuildContext context) {
    final monthKeys = _byMonth.keys.toList();
    return Stack(
      children: [
        ListView.builder(
          controller: _scroll,
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 96),
          itemCount: monthKeys.length,
          itemBuilder: (_, i) {
            final key = monthKeys[i];
            final items = _byMonth[key]!;
            final header = DateFormat.yMMMM().format(DateTime.parse('$key-01'));

            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  child: Text(header,
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.w800,
                          )),
                ),
                ...List.generate(items.length, (j) {
                  final s = items[j];
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: Material(
                      elevation: 1,
                      borderRadius: BorderRadius.circular(12),
                      color: Colors.white,
                      child: ListTile(
                        leading: _Thumb(score: s.score),
                        title: Text(
                          '${s.exercise} · ${DateFormat.MMMd().format(s.date)}',
                          style: const TextStyle(fontWeight: FontWeight.w800),
                        ),
                        subtitle: Text('Score ${s.score} · ${s.reps} reps · ${s.weight} kg'),
                        trailing: const Icon(Icons.chevron_right),
                        onTap: () => context.go('/feedback', extra: {
                          'videoPath': null,
                          'videoKey': s.s3Key ?? 'local/mock/${s.id}.mp4',
                          'report': const {'data': []},
                        }),
                      ),
                    ),
                  );
                }),
              ],
            );
          },
        ),

        // Filter FAB
        Positioned(
          right: 16,
          bottom: 16,
          child: FloatingActionButton.small(
            onPressed: () async {
              final res = await showModalBottomSheet<_HistoryFilter>(
                context: context,
                showDragHandle: true,
                builder: (_) => _FilterSheet(),
              );
              if (res != null) {
                debugPrint('analytics: history_filter_apply ${res.toJson()}');
                // In real app, apply filters to repository and refresh list.
                if (mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Filter applied (mock)')),
                  );
                }
              }
            },
            child: const Icon(Icons.filter_list),
          ),
        ),
      ],
    );
  }
}

class _Thumb extends StatelessWidget {
  const _Thumb({required this.score});
  final int score;

  @override
  Widget build(BuildContext context) {
    final color = score >= 85
        ? const Color(0xFF00C853)
        : (score >= 70 ? const Color(0xFFFFC107) : const Color(0xFFE53935));
    return Stack(
      alignment: Alignment.center,
      children: [
        Container(
          width: 56,
          height: 56,
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [Color(0xFF2196F3), Color(0xFF673AB7)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.all(Radius.circular(10)),
          ),
        ),
        Positioned(
          right: 0,
          bottom: 0,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              '$score',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 11,
                fontWeight: FontWeight.w800,
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ),
        ),
      ],
    );
  }
}

/* ============================================================================
 * Filter sheet (mock)
 * ==========================================================================*/

class _HistoryFilter {
  _HistoryFilter({required this.minScore, required this.types});
  final int minScore;
  final List<String> types;

  Map<String, dynamic> toJson() => {'minScore': minScore, 'types': types};
}

class _FilterSheet extends StatefulWidget {
  @override
  State<_FilterSheet> createState() => _FilterSheetState();
}

class _FilterSheetState extends State<_FilterSheet> {
  double _minScore = 60;
  final _types = <String, bool>{
    'squat': true,
    'deadlift': true,
    'bench': false,
    'pullup': false,
  };

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text('Filters', style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800)),
          const SizedBox(height: 12),
          Row(
            children: [
              const Text('Min score'),
              Expanded(
                child: Slider(
                  value: _minScore,
                  min: 0,
                  max: 100,
                  divisions: 20,
                  label: _minScore.round().toString(),
                  onChanged: (v) => setState(() => _minScore = v),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Align(
            alignment: Alignment.centerLeft,
            child: Wrap(
              spacing: 8,
              children: _types.keys.map((k) {
                final selected = _types[k]!;
                return FilterChip(
                  label: Text(k),
                  selected: selected,
                  onSelected: (v) => setState(() => _types[k] = v),
                );
              }).toList(),
            ),
          ),
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            child: FilledButton(
              onPressed: () {
                Navigator.pop(
                  context,
                  _HistoryFilter(
                    minScore: _minScore.round(),
                    types: _types.entries.where((e) => e.value).map((e) => e.key).toList(),
                  ),
                );
              },
              child: const Text('Apply'),
            ),
          ),
        ],
      ),
    );
  }
}

/* ============================================================================
 * MOCK DATA
 * ==========================================================================*/

class _Session {
  _Session({
    required this.id,
    required this.date,
    required this.exercise,
    required this.score,
    required this.reps,
    required this.weight,
    this.s3Key,
  });

  final String id;
  final DateTime date;
  final String exercise;
  final int score;
  final int reps;
  final int weight;
  final String? s3Key;
}

class _Badge {
  _Badge(this.title, this.short, this.icon, this.bg, this.date);
  final String title;
  final String short;
  final IconData icon;
  final Color bg;
  final DateTime date;
}

class _CoachTip {
  _CoachTip(this.title, this.subtitle, this.icon, this.color);
  final String title;
  final String subtitle;
  final IconData icon;
  final Color color;
}

class _MockData {
  static final _rand = math.Random(7);

  static List<_Session> generateSessions() {
    final now = DateTime.now();
    final list = <_Session>[];
    for (int i = 0; i < 70; i++) {
      final d = now.subtract(Duration(days: i));
      // 40% chance of a session that day
      if (_rand.nextDouble() < 0.4) {
        final n = 1 + _rand.nextInt(2);
        for (int k = 0; k < n; k++) {
          final ex = ['squat', 'deadlift', 'bench', 'pullup'][_rand.nextInt(4)];
          list.add(_Session(
            id: '${d.millisecondsSinceEpoch}-$k',
            date: d.add(Duration(hours: 6 + _rand.nextInt(12))),
            exercise: ex,
            score: 60 + _rand.nextInt(41),
            reps: 5 + _rand.nextInt(8),
            weight: 40 + _rand.nextInt(80),
            s3Key: null,
          ));
        }
      }
    }
    // newest first
    list.sort((a, b) => b.date.compareTo(a.date));
    return list;
  }

  static _Badge get latestBadge => _Badge(
        'Depth Master',
        'Depth',
        Icons.workspace_premium,
        const Color(0xFF00C6A2),
        DateTime.now().subtract(const Duration(days: 3)),
      );

  static List<_Badge> get allBadges => [
        _Badge('Depth Master', 'Depth', Icons.workspace_premium, const Color(0xFF00C6A2),
            DateTime.now()),
        _Badge('Hip Control', 'Hips', Icons.sports_gymnastics, const Color(0xFF7C4DFF),
            DateTime.now()),
        _Badge('Core Stable', 'Core', Icons.shield, const Color(0xFFFF7043),
            DateTime.now()),
        _Badge('Knee Friendly', 'Knees', Icons.accessibility_new, const Color(0xFF29B6F6),
            DateTime.now()),
        _Badge('Consistent', 'Streak', Icons.bolt, const Color(0xFFFFC107),
            DateTime.now()),
        _Badge('Powerhouse', 'Power', Icons.fitness_center, const Color(0xFFE91E63),
            DateTime.now()),
        _Badge('Endurance', 'Endur', Icons.directions_run, const Color(0xFF26A69A),
            DateTime.now()),
        _Badge('Focus', 'Focus', Icons.visibility, const Color(0xFFAB47BC),
            DateTime.now()),
      ];

  static List<_CoachTip> get coachTips => [
        _CoachTip('Drive through heels', 'Keeps bar path vertical', Icons.directions_run,
            const Color(0xFFFF7043)),
        _CoachTip('Brace the core', 'Inhale, lock, then descend', Icons.shield,
            const Color(0xFF26A69A)),
        _CoachTip('Neutral spine', 'Head and hips aligned', Icons.swap_calls,
            const Color(0xFF29B6F6)),
      ];

  static String get reelThumb => 'local/placeholder';
}
