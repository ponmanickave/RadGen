from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import os

# --- BRANDING COLORS ---
PRIMARY_TEAL = HexColor("#13c8ec")
DARK_NAVY = HexColor("#101f22")
SLATE_GREY = HexColor("#64748b")
SOFT_GHOST = HexColor("#f8fafc")

class PDFGenerator:
    def __init__(self):
        self.filename = ""

    def create_report(self, report_obj, output_path):
        c = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4

        # --- PAGE 1: PROFESSIONAL COVER ---
        self._draw_background(c, width, height)
        self._add_modern_header(c, width, height, report_obj)
        
        # Cover Title
        c.setFillColor(DARK_NAVY)
        c.setFont("Helvetica-Bold", 32)
        c.drawString(1*inch, height - 3*inch, "Diagnostic")
        c.setFillColor(PRIMARY_TEAL)
        c.drawString(1*inch, height - 3.5*inch, "Clinical Summary")
        
        # Decorative Element
        c.setStrokeColor(PRIMARY_TEAL)
        c.setLineWidth(3)
        c.line(1*inch, height - 3.8*inch, 2*inch, height - 3.8*inch)
        
        # Patient Card
        self._draw_rounded_rect(c, 1*inch, height - 7.5*inch, width - 2*inch, 3*inch, radius=15, fill_color=colors.white)
        
        c.setFillColor(DARK_NAVY)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1.4*inch, height - 5*inch, "PATIENT PROFILE")
        
        # Grid layout for details
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(SLATE_GREY)
        y = height - 5.5*inch
        
        detail_items = [
            ("FULL NAME", getattr(report_obj, 'patient_name', 'Not Registered')),
            ("IDENTIFIER", f"PT-{report_obj.patient_id}"),
            ("GENERATED ON", report_obj.timestamp.strftime('%B %d, %Y | %I:%M %p')),
            ("SYSTEM STATUS", report_obj.status.upper())
        ]
        
        for label, value in detail_items:
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(SLATE_GREY)
            c.drawString(1.4*inch, y, label)
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(DARK_NAVY)
            c.drawString(1.4*inch, y - 0.2*inch, str(value))
            y -= 0.6*inch

        self._add_modern_footer(c, width, height, page_num=1)
        c.showPage() 

        # --- PAGE 2: ANALYSIS & FINDINGS ---
        self._draw_background(c, width, height)
        self._add_modern_header(c, width, height, report_obj)
        
        c.setFillColor(DARK_NAVY)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(1*inch, height - 2*inch, "Neural Analysis Findings")
        
        # Main Findings container
        y_pos = height - 2.5*inch
        findings = report_obj.prediction_data.get('findings', {})
        
        if findings:
            for section, text in findings.items():
                # Section Box
                self._draw_rounded_rect(c, 1*inch, y_pos - 0.7*inch, width - 2*inch, 0.8*inch, radius=10, fill_color=SOFT_GHOST)
                
                c.setFont("Helvetica-Bold", 10)
                c.setFillColor(PRIMARY_TEAL)
                c.drawString(1.3*inch, y_pos - 0.2*inch, section.upper())
                
                c.setFont("Helvetica", 11)
                c.setFillColor(DARK_NAVY)
                c.drawString(1.3*inch, y_pos - 0.45*inch, text)
                y_pos -= 1*inch
        
        # Summary Section (Bottom)
        y_pos = 3.5*inch
        c.setFillColor(DARK_NAVY)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1*inch, y_pos, "AI Clinical Impression")
        
        # Impression Card
        imp_color = HexColor("#fee2e2") if report_obj.status == "Malignant" else HexColor("#dcfce7")
        self._draw_rounded_rect(c, 1*inch, y_pos - 2*inch, width - 2*inch, 1.8*inch, radius=12, fill_color=imp_color)
        
        c.setFillColor(DARK_NAVY)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(1.3*inch, y_pos - 0.4*inch, "SUMMARY STATEMENT:")
        c.setFont("Helvetica", 11)
        c.drawString(1.3*inch, y_pos - 0.65*inch, report_obj.prediction_data.get('impression', 'Correlation advised.'))
        
        c.setFont("Helvetica-Bold", 11)
        c.drawString(1.3*inch, y_pos - 1.1*inch, "RECOMMENDED ACTION:")
        c.setFont("Helvetica-Oblique", 11)
        c.drawString(1.3*inch, y_pos - 1.35*inch, report_obj.prediction_data.get('recommendation', 'Consult radiologist.'))

        self._add_modern_footer(c, width, height, page_num=2)
        c.save()

    def _draw_background(self, c, width, height):
        # Subtle gradient-like background or just plain off-white for printing
        c.setFillColor(SOFT_GHOST)
        c.rect(0, 0, width, height, fill=1, stroke=0)

    def _add_modern_header(self, c, width, height, report_obj):
        # Dark Header Bar
        c.setFillColor(DARK_NAVY)
        c.rect(0, height - 1.2*inch, width, 1.2*inch, fill=1, stroke=0)
        
        # Logo Block
        c.setFillColor(PRIMARY_TEAL)
        c.roundRect(0.5*inch, height - 0.9*inch, 0.4*inch, 0.4*inch, 8, fill=1, stroke=0)
        c.setFillColor(DARK_NAVY)
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(0.7*inch, height - 0.78*inch, "R")
        
        # Brand Text
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(1.1*inch, height - 0.75*inch, "RadGen")
        c.setFont("Helvetica", 8)
        c.setFillColor(PRIMARY_TEAL)
        c.drawString(1.1*inch, height - 0.9*inch, "NEURAL VISION ENGINE")
        
        # Right aligned info
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 8)
        c.drawRightString(width - 0.5*inch, height - 0.6*inch, "CLINICAL FINDINGS REPORT")
        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(PRIMARY_TEAL)
        c.drawRightString(width - 0.5*inch, height - 0.9*inch, f"#{report_obj.id}")

    def _add_modern_footer(self, c, width, height, page_num):
        # Subtle footer line
        c.setStrokeColor(HexColor("#e2e8f0"))
        c.setLineWidth(1)
        c.line(0.5*inch, 0.8*inch, width - 0.5*inch, 0.8*inch)
        
        c.setFillColor(SLATE_GREY)
        c.setFont("Helvetica", 8)
        c.drawString(0.5*inch, 0.6*inch, "Electronically Certified by RadGen Deep Health Network")
        c.drawRightString(width - 0.5*inch, 0.6*inch, f"Page {page_num} of 2")
        
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(width/2, 0.4*inch, "THIS IS AN AI-AUGMENTED DOCUMENT. FINAL REVIEW BY MEDICAL PROFESSIONAL IS REQUIRED.")

    def _draw_rounded_rect(self, c, x, y, w, h, radius, fill_color):
        c.setFillColor(fill_color)
        c.setStrokeColor(HexColor("#e2e8f0"))
        c.setLineWidth(0.5)
        c.roundRect(x, y, w, h, radius, fill=1, stroke=1)
