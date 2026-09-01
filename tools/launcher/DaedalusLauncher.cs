using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

[assembly: AssemblyTitle("Daedalus AI Engineering Suite")]
[assembly: AssemblyDescription("Local-first neural-network engineering workbench")]
[assembly: AssemblyCompany("Daedalus Contributors")]
[assembly: AssemblyProduct("Daedalus AI Engineering Suite")]
[assembly: AssemblyCopyright("Copyright Daedalus contributors")]
[assembly: AssemblyVersion("0.1.0.0")]
[assembly: AssemblyFileVersion("0.1.0.0")]

internal static class Program
{
    private const string ProductName = "Daedalus AI Engineering Suite";

    [STAThread]
    private static int Main()
    {
        string root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(
            Path.DirectorySeparatorChar,
            Path.AltDirectorySeparatorChar
        );
        string python = Path.Combine(root, ".venv", "Scripts", "pythonw.exe");

        try
        {
            if (!File.Exists(python) && !InstallRuntime(root, python))
            {
                return 1;
            }

            ProcessStartInfo application = new ProcessStartInfo();
            application.FileName = python;
            application.Arguments = "-m daedalus";
            application.WorkingDirectory = root;
            application.UseShellExecute = false;
            application.CreateNoWindow = true;
            Process.Start(application);
            return 0;
        }
        catch (Exception error)
        {
            ShowError("Daedalus could not start.\n\n" + error.Message);
            return 1;
        }
    }

    private static bool InstallRuntime(string root, string python)
    {
        string installer = Path.Combine(root, "Install-Daedalus.bat");
        if (!File.Exists(installer))
        {
            ShowError(
                "The Daedalus runtime is not installed and Install-Daedalus.bat " +
                "is missing beside this launcher."
            );
            return false;
        }

        ProcessStartInfo setup = new ProcessStartInfo();
        setup.FileName = installer;
        setup.Arguments = "-NoLaunch";
        setup.WorkingDirectory = root;
        setup.UseShellExecute = true;

        Process process = Process.Start(setup);
        if (process == null)
        {
            ShowError("Windows could not start the Daedalus installer.");
            return false;
        }
        process.WaitForExit();
        if (process.ExitCode != 0 || !File.Exists(python))
        {
            ShowError(
                "Daedalus installation did not complete. Run Install-Daedalus.bat " +
                "to review the installation details."
            );
            return false;
        }
        return true;
    }

    private static void ShowError(string message)
    {
        MessageBox.Show(
            message,
            ProductName,
            MessageBoxButtons.OK,
            MessageBoxIcon.Error
        );
    }
}
