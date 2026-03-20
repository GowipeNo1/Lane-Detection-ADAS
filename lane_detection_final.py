import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # 1. SETUP & VIDEO SAVING INITIALIZATION
    model = YOLO(r'F:\Desktop\Lane detection project\runs\detect\train4\weights\best.pt')
    cap = cv2.VideoCapture(r'F:\Desktop\Lane detection project\test_video.mp4')
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Fetch properties for the output video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    output_path = r'F:\Desktop\Lane detection project\final_adas_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Lane history buffer for anti-flicker stability
    history_l, history_r = [], []
    buffer_size = 20 

    print(f"Processing started. Saving to: {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 2. DETECTION
        # Low confidence threshold captures distant markings
        results = model(frame, conf=0.10, verbose=False)
        curr_l, curr_r = None, None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2
                pts = np.array([[cx, y2], [cx, (y1+y2)/2], [cx, y1]], np.float32)
                
                if cx < w / 2: curr_l = pts
                else:          curr_r = pts

        # 3. BUFFER UPDATES (Persistence Logic)
        if curr_l is not None:
            history_l.append(curr_l)
            if len(history_l) > buffer_size: history_l.pop(0)
        if curr_r is not None:
            history_r.append(curr_r)
            if len(history_r) > buffer_size: history_r.pop(0)

        # 4. RENDERING ADAS GRAPHICS
        if len(history_l) > 0 and len(history_r) > 0:
            try:
                # Average the buffer for smooth constant curves
                avg_l = np.mean(history_l, axis=0)
                avg_r = np.mean(history_r, axis=0)
                
                # 2nd-degree polynomial fit
                poly_l = np.polyfit(avg_l[:, 1], avg_l[:, 0], 2)
                poly_r = np.polyfit(avg_r[:, 1], avg_r[:, 0], 2)
                
                # Horizon at 48% height for depth
                plot_y = np.linspace(int(h * 0.48), h, 35)
                curve_l = np.vstack([np.polyval(poly_l, plot_y), plot_y]).T.astype(np.int32)
                curve_r = np.vstack([np.polyval(poly_r, plot_y), plot_y]).T.astype(np.int32)

                overlay = frame.copy()
                # 3D Mesh Rungs
                for i in range(0, len(curve_l), 5):
                    cv2.line(overlay, tuple(curve_l[i]), tuple(curve_r[i]), (255, 255, 255), 1)

                # Safe Path Polygon
                poly_pts = np.vstack([curve_l, curve_r[::-1]])
                cv2.fillPoly(overlay, [poly_pts], (0, 255, 100))
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                # Ego-Vehicle marker following lane center
                mx, my = (curve_l[-1][0] + curve_r[-1][0]) // 2, (curve_l[-1][1] + curve_r[-1][1]) // 2
                cv2.circle(frame, (mx, my - 20), 15, (0, 255, 255), -1)
                
                cv2.polylines(frame, [curve_l], False, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.polylines(frame, [curve_r], False, (0, 0, 255), 4, cv2.LINE_AA)
            except: pass

        # 5. STATUS HUD (Corrected l_stat and r_stat)
        cv2.rectangle(frame, (10, 10), (280, 90), (30, 30, 30), -1)
        
        # Defining status colors to fix NameError
        l_stat_color = (0, 255, 0) if curr_l is not None else (0, 165, 255)
        r_stat_color = (0, 255, 0) if curr_r is not None else (0, 165, 255)
        
        cv2.putText(frame, f"L-LANE: {'ACTIVE' if curr_l is not None else 'MEMORY'}", 
                    (20, 45), 0, 0.6, l_stat_color, 2)
        cv2.putText(frame, f"R-LANE: {'ACTIVE' if curr_r is not None else 'MEMORY'}", 
                    (20, 75), 0, 0.6, r_stat_color, 2)

        # 6. OUTPUT & DISPLAY
        out.write(frame)
        cv2.imshow("ADAS Final Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # CLEANUP
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Success! Video saved at: {output_path}")

if __name__ == '__main__':
    main()