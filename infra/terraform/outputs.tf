output "eep_url" {
  description = "Public URL of the EEP Cloud Run service"
  value       = google_cloud_run_v2_service.eep.uri
}

output "iep2_url" {
  description = "Public URL of the IEP2 Cloud Run service"
  value       = google_cloud_run_v2_service.iep2.uri
}

output "iep3_url" {
  description = "Public URL of the IEP3 Cloud Run service"
  value       = google_cloud_run_v2_service.iep3.uri
}

output "iep4_url" {
  description = "Public URL of the IEP4 Cloud Run service"
  value       = google_cloud_run_v2_service.iep4.uri
}

output "db_instance_ip" {
  description = "Internal IP of the TimescaleDB+Redis GCE instance (if created)"
  value       = var.enable_db_instance ? google_compute_instance.db[0].network_interface[0].network_ip : null
}

output "db_instance_external_ip" {
  description = "External IP of the TimescaleDB+Redis GCE instance (if created)"
  value       = var.enable_db_instance ? google_compute_instance.db[0].network_interface[0].access_config[0].nat_ip : null
}
