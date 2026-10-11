import io


def build_prediction_report(record, image_bytes=None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    output = io.BytesIO(); styles = getSampleStyleSheet()
    story = [Paragraph("CerviScan AI — Screening Report", styles["Title"]), Spacer(1, 12)]
    if image_bytes:
        try:
            uploaded_image = Image(io.BytesIO(image_bytes), width=180, height=135, kind="proportional")
            story += [Paragraph("Uploaded image", styles["Heading2"]), uploaded_image, Spacer(1, 12)]
        except Exception:
            # Keep the report available if a legacy image cannot be embedded.
            pass
    details = [
        ["Patient / Sample ID", record["patient_id"]],
        ["Date/time", record["created_at"]],
        ["Image", record["image_filename"]],
        ["Cells Localized", str(record.get("cells_detected_count", "1 (Full Image)"))],
        ["Primary Cell Class", record["predicted_class"]],
        ["Model confidence", f'{record["confidence"] * 100:.2f}%'],
        ["Risk/status", record["risk_status"]],
        ["Model", record["model_name"]],
        ["Processing time", f'{record["processing_time_ms"]} ms']
    ]
    table = Table(details, colWidths=[150, 350]); table.setStyle(TableStyle([("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eef2ff")), ("GRID", (0,0), (-1,-1), .25, colors.grey), ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"), ("VALIGN", (0,0), (-1,-1), "TOP"), ("PADDING", (0,0), (-1,-1), 7)])); story.append(table)
    story += [Spacer(1, 15), Paragraph("Class probabilities", styles["Heading2"])]
    probabilities = [["Class", "Probability"]] + [[label, f"{value * 100:.2f}%"] for label, value in record["class_probabilities"].items()]
    table = Table(probabilities, colWidths=[300, 200]); table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4f46e5")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), .25, colors.grey), ("PADDING", (0,0), (-1,-1), 7)])); story.append(table)
    story += [Spacer(1, 15), Paragraph("Interpretation", styles["Heading2"]), Paragraph("This is an AI-assisted classification for screening support. It is not a definitive medical diagnosis. A qualified clinician/pathologist must review and confirm all results.", styles["BodyText"])]
    SimpleDocTemplate(output, pagesize=A4, title="CerviScan AI report").build(story); output.seek(0); return output
