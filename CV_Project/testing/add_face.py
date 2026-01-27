import cv2
import os

# ==========================================
# SCRIPT 1: MANUAL DATA COLLECTION
# ==========================================

def manual_registration():
    print("--- FACE REGISTRATION SYSTEM ---")
    name = input("Enter the user name (e.g., John): ").strip()
    
    if not name:
        print("Name cannot be empty.")
        return

    # Create a specific folder for this user
    base_dir = "dataset"
    user_path = os.path.join(base_dir, name)
    
    if not os.path.exists(user_path):
        os.makedirs(user_path)
        print(f"Created folder: {user_path}")
    else:
        print(f"Folder exists. Appending new images to: {user_path}")

    # Count existing files so we don't overwrite them
    existing_files = len(os.listdir(user_path))
    count = existing_files

    cap = cv2.VideoCapture(0)
    
    print("\nINSTRUCTIONS:")
    print("1. Turn your head slightly Left, Right, Up, and Down.")
    print("2. Press 's' to SAVE a photo at that angle.")
    print("3. Press 'q' to QUIT when finished (aim for 20-50 photos).")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Camera not found.")
            break

        # Display the frame
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Images Saved: {count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 's' to Save, 'q' to Quit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        cv2.imshow("Registration Mode", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # --- SAVE LOGIC ---
        if key == ord('s'):
            count += 1
            filename = f"{name}_{count}.jpg"
            file_path = os.path.join(user_path, filename)
            
            cv2.imwrite(file_path, frame)
            print(f"[Saved] {filename}")
            
            # Visual feedback (flash effect)
            cv2.imshow("Registration Mode", 255 - display_frame)
            cv2.waitKey(50)

        # --- QUIT LOGIC ---
        elif key == ord('q'):
            print(f"\nRegistration finished. Total images for {name}: {count}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    manual_registration()