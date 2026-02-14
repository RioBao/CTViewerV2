Style Guide — Industrial 3D Viewer UI
🎯 Design Philosophy

Content first, controls second
Reduce visual clutter, emphasize the data and viewports. Every UI element should exist to support user focus and reduce cognitive load.

This guide uses:

Minimal chrome

High hierarchy clarity

Dark, technical aesthetic

Focused interaction cues

🎨 Color Palette
Core Palette
Role	Example	Usage
Background (canvas)	#111318	Main viewport/ background
Panel surface	#1A1D24	Inspector, side docks
Separator / subtle borders	#2A2F3A	Section separators
Accent (primary)	#006DFE	Active controls, highlights
Accent (secondary)	#FFC857	Alerts, warnings
Text high contrast	#FFFFFF	Main labels
Text medium contrast	#8A8F98	Labels, secondary text

Best practices

Only use accent colors for interactive elements.

Maintain high contrast for readability.

✍ Typography
Font Choices

Primary font: Inter / IBM Plex Sans / Roboto

Weights:

Regular (400) for body text

SemiBold (600) for labels/buttons

Bold (700) for headers

Core Text Styles
Usage	Font Size	Weight
Header (H1)	20–24pt	700
Section Title	16–18pt	600
Body Text / Labels	12–14pt	400–600
Micro labels	10pt	400

Notes

Use consistent spacing to create visual hierarchy.

Maintain comfortable line spacing for readability.

📐 Spacing & Layout
Base Grid

Use an 8px grid for spacing (margins, padding, element gaps).

Examples:

8px — minimal spacing
16px — panel padding
24px — group separation
32px — section separation


Consistent spacing:

Improves visual flow

Reduces clutter

Helps align elements into predictable structure

🧠 Visual Hierarchy

Use:

Typeface weight

Color contrast

Size differences

Spacing

to guide attention.

Rule of Thumb

Viewports (largest)

Mode indicators

Active controls

Secondary controls

Status text

🛠 Component Styles
Buttons

Height: 32px

Border-radius: 8px

Background: accent or neutral

Text uppercase for clarity

Disabled state: 40% opacity

Tabs / Mode Selectors

Simple underlines

No heavy shadows

Use accent color for active state

Iconography

Line icons, minimal stroke

20–24px for main toolbar

16–20px for inspector actions

🧊 Panels & Surfaces
Panel Style

Rounded corners (8px)

Soft inner shadows, subtle depth

Slight translucency optional (glassmorphism)

Example:

background: rgba(26,29,36,0.95)
backdrop-filter: blur(10px)


This gives a floating feel while keeping focus on content.

✨ Interactions & States
Active / Hover

Active → accent highlight

Hover → slight brightness increase

Pressed → darken slightly

Disabled

Lower opacity

No hover state

Micro interactions (transitions ~100–150ms) are key for perceived polish.

🧪 UX Rules to Follow

Hierarchy: Use size/contrast to show importance.

Minimize cognitive load: Avoid too many simultaneous controls.

Design grayscale first: Focus on structure before color.

Flat/minimal UI: Simple shapes, minimal gradients, clean icons.

📌 Example Tokens (JSON Style)
{
  "colors": {
    "background": "#111318",
    "panel": "#1A1D24",
    "separator": "#2A2F3A",
    "primary": "#006DFE",
    "secondary": "#FFC857",
    "textHigh": "#FFFFFF",
    "textMid": "#8A8F98"
  },
  "typography": {
    "fontFamily": "Inter, sans-serif",
    "h1": { "size": "24px", "weight": "700" },
    "body": { "size": "14px", "weight": "400" },
    "label": { "size": "12px", "weight": "600" }
  },
  "spacing": {
    "base": "8px",
    "medium": "16px",
    "large": "24px"
  }
}