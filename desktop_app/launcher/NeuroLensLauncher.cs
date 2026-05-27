using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class Program
{
    [STAThread]
    private static int Main()
    {
        var workspace = @"C:\Users\ABC\Desktop\ai_asistan_workspace6 - app";
        var batchPath = Path.Combine(workspace, "Start_Desktop_App_OneClick.bat");

        if (!File.Exists(batchPath))
        {
            MessageBox.Show(
                "Start_Desktop_App_OneClick.bat not found.\n\nExpected location:\n" + batchPath,
                "NeuroLens Clinical Workspace",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
            return 1;
        }

        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "cmd.exe",
                Arguments = "/c start \"\" /min \"" + batchPath + "\"",
                WorkingDirectory = workspace,
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden
            };

            Process.Start(psi);
            return 0;
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                "Application could not be started.\n\n" + ex.Message,
                "NeuroLens Clinical Workspace",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
            return 1;
        }
    }
}
