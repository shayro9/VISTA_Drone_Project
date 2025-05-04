using System.IO;
using UnityEngine;

public class PoseLogger : MonoBehaviour
{
    public int frameInterval = 5;  // Capture every N frames
    private int frameCount = 0;
    private StreamWriter writer;
    public string filePath;
    private Vector3 start_pos;
    private Quaternion start_rot;

    void Start()
    {
	Camera cam = Camera.main;
	float fov = cam.fieldOfView;
	float width = cam.pixelWidth;
	float height = cam.pixelHeight;

	float fy = 0.5f * height / Mathf.Tan(0.5f * fov * Mathf.Deg2Rad);
	float fx = fy;
	float cx = width / 2.0f;
	float cy = height / 2.0f;

        start_pos = transform.position;
	start_rot = transform.rotation;    

	Debug.Log(fx + " " + fy + " " + cx + " " + cy);
        // Set file path to project folder
        filePath = Path.Combine(filePath, "pose_log.txt");

        // Open writer and create/overwrite file
        writer = new StreamWriter(filePath, false);
        writer.Flush();
    }

    void Update()
    {
        if (frameCount % frameInterval == 0)
        {
            Vector3 pos = transform.position - start_pos;
            Quaternion rot = transform.rotation;
	        Quaternion relative = Quaternion.Inverse(start_rot) * rot;


            string log = $"{frameCount / 5} {pos.x:F3} {pos.y:F3} {pos.z:F3} {relative.x:F3} {relative.y:F3} {relative.z:F3} {relative.w:F3} \n";
            writer.WriteLine(log);
            writer.Flush(); // optional, ensures it's written immediately
        }
	frameCount++;
    }

    void OnApplicationQuit()
    {
        if (writer != null)
        {
            writer.Close();
        }
    }
}