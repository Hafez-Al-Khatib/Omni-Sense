# ──────────────────────────────────────────────────────────────────────────────
# Omni-Sense GCP Infrastructure — Cloud Run + GCE DB Box
# Student demo: minimal, cost-optimized, publicly accessible
# ──────────────────────────────────────────────────────────────────────────────

locals {
  services = {
    eep = {
      name          = "eep"
      image         = "${var.image_registry}/omni-sense-eep:${var.image_tag}"
      port          = 8000
      min_instances = 0
      max_instances = 5
      cpu           = "1"
      memory        = "512Mi"
      env = {
        OMNI_IEP2_URL = "${google_cloud_run_v2_service.iep2.uri}"
        OMNI_IEP3_URL = "${google_cloud_run_v2_service.iep3.uri}"
        OMNI_IEP4_URL = "${google_cloud_run_v2_service.iep4.uri}"
        OMNI_RATE_LIMIT = "5/minute"
      }
    }
    iep2 = {
      name          = "iep2"
      image         = "${var.image_registry}/omni-sense-iep2:${var.image_tag}"
      port          = 8002
      min_instances = 0
      max_instances = 5
      cpu           = "1"
      memory        = "512Mi"
      env           = {}
    }
    iep3 = {
      name          = "iep3"
      image         = "${var.image_registry}/omni-sense-iep3:${var.image_tag}"
      port          = 8003
      min_instances = 0
      max_instances = 5
      cpu           = "1"
      memory        = "512Mi"
      env           = {}
    }
    iep4 = {
      name          = "iep4"
      image         = "${var.image_registry}/omni-sense-iep4:${var.image_tag}"
      port          = 8004
      min_instances = 1   # avoid cold-start SLO breach
      max_instances = 5
      cpu           = "2"
      memory        = "2Gi"
      env           = {}
    }
  }
}

# ── Enable APIs ───────────────────────────────────────────────────────────────
resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "compute" {
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

# ── Artifact Registry (optional, can use Docker Hub) ──────────────────────────
resource "google_artifact_registry_repository" "omni_sense" {
  location      = var.gcp_region
  repository_id = "omni-sense"
  format        = "DOCKER"
  description   = "Omni-Sense container images"
}

# ── Cloud Run Services ────────────────────────────────────────────────────────
resource "google_cloud_run_v2_service" "eep" {
  name     = "omni-sense-eep"
  location = var.gcp_region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instances = local.services.eep.min_instances
      max_instances = local.services.eep.max_instances
    }
    containers {
      image = local.services.eep.image
      ports {
        container_port = local.services.eep.port
      }
      resources {
        limits = {
          cpu    = local.services.eep.cpu
          memory = local.services.eep.memory
        }
      }
      dynamic "env" {
        for_each = local.services.eep.env
        content {
          name  = env.key
          value = env.value
        }
      }
    }
  }

  depends_on = [google_project_service.run]
}

resource "google_cloud_run_v2_service" "iep2" {
  name     = "omni-sense-iep2"
  location = var.gcp_region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instances = local.services.iep2.min_instances
      max_instances = local.services.iep2.max_instances
    }
    containers {
      image = local.services.iep2.image
      ports {
        container_port = local.services.iep2.port
      }
      resources {
        limits = {
          cpu    = local.services.iep2.cpu
          memory = local.services.iep2.memory
        }
      }
      startup_probe {
        initial_delay_seconds = 5
        timeout_seconds       = 5
        period_seconds        = 5
        failure_threshold     = 12
        http_get {
          path = "/health"
          port = local.services.iep2.port
        }
      }
    }
  }

  depends_on = [google_project_service.run]
}

resource "google_cloud_run_v2_service" "iep3" {
  name     = "omni-sense-iep3"
  location = var.gcp_region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instances = local.services.iep3.min_instances
      max_instances = local.services.iep3.max_instances
    }
    containers {
      image = local.services.iep3.image
      ports {
        container_port = local.services.iep3.port
      }
      resources {
        limits = {
          cpu    = local.services.iep3.cpu
          memory = local.services.iep3.memory
        }
      }
      startup_probe {
        initial_delay_seconds = 5
        timeout_seconds       = 5
        period_seconds        = 5
        failure_threshold     = 12
        http_get {
          path = "/health"
          port = local.services.iep3.port
        }
      }
    }
  }

  depends_on = [google_project_service.run]
}

resource "google_cloud_run_v2_service" "iep4" {
  name     = "omni-sense-iep4"
  location = var.gcp_region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instances = local.services.iep4.min_instances
      max_instances = local.services.iep4.max_instances
    }
    containers {
      image = local.services.iep4.image
      ports {
        container_port = local.services.iep4.port
      }
      resources {
        limits = {
          cpu    = local.services.iep4.cpu
          memory = local.services.iep4.memory
        }
      }
      startup_probe {
        initial_delay_seconds = 10
        timeout_seconds       = 10
        period_seconds        = 10
        failure_threshold     = 18
        http_get {
          path = "/health"
          port = local.services.iep4.port
        }
      }
    }
  }

  depends_on = [google_project_service.run]
}

# ── Allow unauthenticated invocations (demo only) ─────────────────────────────
resource "google_cloud_run_v2_service_iam_member" "eep_public" {
  location = var.gcp_region
  name     = google_cloud_run_v2_service.eep.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "iep2_public" {
  location = var.gcp_region
  name     = google_cloud_run_v2_service.iep2.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "iep3_public" {
  location = var.gcp_region
  name     = google_cloud_run_v2_service.iep3.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "iep4_public" {
  location = var.gcp_region
  name     = google_cloud_run_v2_service.iep4.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ── GCE e2-small instance for TimescaleDB + Redis ─────────────────────────────
resource "google_compute_instance" "db" {
  count        = var.enable_db_instance ? 1 : 0
  name         = var.db_instance_name
  machine_type = "e2-small"
  zone         = var.gcp_zone

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 20
    }
  }

  network_interface {
    network = "default"
    access_config {} # ephemeral external IP for demo
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    set -e
    apt-get update
    apt-get install -y docker.io docker-compose
    # Run TimescaleDB
    docker run -d --name timescaledb \
      -e POSTGRES_USER=omni \
      -e POSTGRES_PASSWORD=changeme \
      -e POSTGRES_DB=omnisense \
      -p 5432:5432 \
      -v timescale-data:/var/lib/postgresql/data \
      timescale/timescaledb:latest-pg16
    # Run Redis
    docker run -d --name redis \
      -p 6379:6379 \
      redis:7.2-alpine \
      redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
  EOT

  tags = ["omni-sense-db"]

  depends_on = [google_project_service.compute]
}

# ── Firewall rule to allow DB access from Cloud Run (demo) ────────────────────
resource "google_compute_firewall" "db_allow_cloudrun" {
  count       = var.enable_db_instance ? 1 : 0
  name        = "allow-cloudrun-to-db"
  network     = "default"
  description = "Allow Cloud Run (via serverless VPC connector or public IP) to reach DB box"

  allow {
    protocol = "tcp"
    ports    = ["5432", "6379"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["omni-sense-db"]
}
