// Forja3D Bridge: lista tus modelos de Forja3D, genera nuevos desde texto y los importa a Assets/Forja3D.
// Menú: Window > Forja3D. FBX (con Blender instalado en Forja3D) conserva el esqueleto Humanoid y las animaciones;
// GLB necesita el paquete glTFast (com.unity.cloud.gltfast).
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;
using UnityEngine.Networking;

namespace Forja3D
{
    public class Forja3DWindow : EditorWindow
    {
        const string DefaultServer = "http://127.0.0.1:8000";
        const string ServerKey = "Forja3D.Server";

        [Serializable] class Job { public string id; public string title; public string mode; public string status; public int progress; public string stage; public string error; public string thumbnail_url; }
        [Serializable] class JobList { public List<Job> items = new List<Job>(); }
        [Serializable] class TextBody { public string prompt; public string quality; public string pose; public int variants = 1; }

        string server;
        string prompt = "";
        int quality = 2;
        bool aPose = true;
        bool useFbx = true;
        string status = "";
        Vector2 scroll;
        readonly List<Job> jobs = new List<Job>();
        double nextRefresh;

        [MenuItem("Window/Forja3D")]
        static void Open() => GetWindow<Forja3DWindow>("Forja3D").Show();

        void OnEnable()
        {
            server = EditorPrefs.GetString(ServerKey, DefaultServer);
            EditorApplication.update += Tick;
            Refresh();
        }

        void OnDisable() => EditorApplication.update -= Tick;

        void Tick()
        {
            // los modelos en curso se vuelven a consultar cada par de segundos
            if (EditorApplication.timeSinceStartup < nextRefresh) return;
            nextRefresh = EditorApplication.timeSinceStartup + 2.5;
            if (jobs.Exists(j => j.status == "queued" || j.status == "running")) Refresh();
        }

        string Url(string path) => server.TrimEnd('/') + path;

        void OnGUI()
        {
            EditorGUI.BeginChangeCheck();
            server = EditorGUILayout.TextField("Servidor", server);
            if (EditorGUI.EndChangeCheck()) EditorPrefs.SetString(ServerKey, server);

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Crear desde texto", EditorStyles.boldLabel);
            prompt = EditorGUILayout.TextField("Descripción", prompt);
            quality = EditorGUILayout.Popup("Calidad", quality, new[] { "Borrador", "Estándar", "Full HD" });
            aPose = EditorGUILayout.Toggle("Pose A (para rig)", aPose);
            using (new EditorGUI.DisabledScope(prompt.Trim().Length < 3))
                if (GUILayout.Button("Generar modelo 3D")) Generate();

            EditorGUILayout.Space();
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Mis modelos", EditorStyles.boldLabel);
            useFbx = EditorGUILayout.ToggleLeft("FBX (si no, GLB)", useFbx, GUILayout.Width(140));
            if (GUILayout.Button("Actualizar", GUILayout.Width(80))) Refresh();
            EditorGUILayout.EndHorizontal();

            scroll = EditorGUILayout.BeginScrollView(scroll);
            foreach (var job in jobs)
            {
                EditorGUILayout.BeginHorizontal("box");
                EditorGUILayout.LabelField(job.title, GUILayout.MinWidth(120));
                var state = job.status == "running" ? job.progress + "%" : job.status == "succeeded" ? "listo" : job.status;
                EditorGUILayout.LabelField(state, GUILayout.Width(70));
                using (new EditorGUI.DisabledScope(job.status != "succeeded"))
                    if (GUILayout.Button("Importar", GUILayout.Width(70))) Import(job);
                EditorGUILayout.EndHorizontal();
            }
            EditorGUILayout.EndScrollView();
            if (!string.IsNullOrEmpty(status)) EditorGUILayout.HelpBox(status, MessageType.None);
        }

        void Refresh()
        {
            var req = UnityWebRequest.Get(Url("/api/jobs"));
            req.SendWebRequest().completed += _ =>
            {
                if (req.result != UnityWebRequest.Result.Success) { status = "Sin conexión con Forja3D: " + req.error; Repaint(); req.Dispose(); return; }
                var list = JsonUtility.FromJson<JobList>("{\"items\":" + req.downloadHandler.text + "}");
                jobs.Clear();
                jobs.AddRange(list.items);
                status = jobs.Count + " modelos";
                Repaint();
                req.Dispose();
            };
        }

        void Generate()
        {
            var body = JsonUtility.ToJson(new TextBody { prompt = prompt.Trim(), quality = new[] { "draft", "standard", "high" }[quality], pose = aPose ? "a-pose" : "none" });
            var req = new UnityWebRequest(Url("/api/jobs/text"), "POST")
            {
                uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body)),
                downloadHandler = new DownloadHandlerBuffer(),
            };
            req.SetRequestHeader("Content-Type", "application/json");
            req.SendWebRequest().completed += _ =>
            {
                status = req.result == UnityWebRequest.Result.Success ? "Generando… aparecerá abajo." : "No se pudo generar: " + req.downloadHandler.text;
                req.Dispose();
                Refresh();
            };
        }

        void Import(Job job)
        {
            var format = useFbx ? "fbx" : "glb";
            var req = UnityWebRequest.Get(Url("/api/jobs/" + job.id + "/export?format=" + format));
            req.SendWebRequest().completed += _ =>
            {
                if (req.result != UnityWebRequest.Result.Success)
                {
                    status = format == "fbx" ? "FBX no disponible (Forja3D necesita Blender): prueba con GLB. " + req.downloadHandler.text : "No se pudo importar: " + req.downloadHandler.text;
                    req.Dispose();
                    Repaint();
                    return;
                }
                var dir = "Assets/Forja3D";
                Directory.CreateDirectory(dir);
                var path = AssetDatabase.GenerateUniqueAssetPath(dir + "/" + Safe(job.title) + "." + format);
                File.WriteAllBytes(path, req.downloadHandler.data);
                AssetDatabase.ImportAsset(path);
                var asset = AssetDatabase.LoadMainAssetAtPath(path);
                if (asset != null) { Selection.activeObject = asset; EditorGUIUtility.PingObject(asset); }
                status = "Importado en " + path + (format == "glb" ? " (necesita glTFast)" : "");
                req.Dispose();
                Repaint();
            };
        }

        static string Safe(string name)
        {
            var sb = new StringBuilder();
            foreach (var c in name) sb.Append(char.IsLetterOrDigit(c) ? c : '_');
            var s = sb.ToString().Trim('_');
            return s.Length == 0 ? "modelo" : (s.Length > 40 ? s.Substring(0, 40) : s);
        }
    }
}
