//! `shatterbox-tools` — offline helpers for the autotest harness.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use shatterbox_tools::{decode_png, diff, write_png};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        print_usage();
        bail!("missing command");
    };
    match command.as_str() {
        "diff" => diff_command(args.collect()),
        "--help" | "-h" | "help" => {
            print_usage();
            Ok(())
        }
        other => {
            print_usage();
            bail!("unknown command `{other}`");
        }
    }
}

fn print_usage() {
    println!("shatterbox-tools <command>\n");
    println!("  diff <a.png> <b.png> [--out diff.png] [--threshold N]");
    println!("      Compares two screenshots and reports the mean absolute");
    println!("      deviation, the changed-pixel fraction and writes a diff PNG.");
}

fn diff_command(args: Vec<String>) -> Result<()> {
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut out = PathBuf::from("diff.png");
    let mut threshold = 2u8;
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--out" => {
                out = PathBuf::from(
                    iter.next()
                        .context("--out needs a path (e.g. --out target/diff.png)")?,
                );
            }
            "--threshold" => {
                threshold = iter
                    .next()
                    .context("--threshold needs a value 0..255")?
                    .parse()
                    .context("--threshold must be an integer 0..255")?;
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            other if other.starts_with("--") => bail!("unknown flag `{other}`"),
            path => paths.push(PathBuf::from(path)),
        }
    }
    if paths.len() != 2 {
        bail!("diff needs exactly two PNG paths");
    }
    let left = decode_png(&paths[0])?;
    let right = decode_png(&paths[1])?;
    let (report, diff_image) = diff(&left, &right, threshold)?;
    write_png(&out, &diff_image)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    println!(
        "mean abs deviation {:.3}/255, {:.2}% of pixels changed, max {} -> {}",
        report.mean_abs,
        report.changed_fraction * 100.0,
        report.max_abs,
        out.display()
    );
    Ok(())
}
