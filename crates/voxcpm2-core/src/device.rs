use candle_core::Device;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    Auto,
    Cpu,
    Cuda(usize),
    Metal(usize),
}

impl DevicePreference {
    pub fn parse(value: &str) -> anyhow::Result<Self> {
        let v = value.to_ascii_lowercase();
        if v == "auto" {
            return Ok(Self::Auto);
        }
        if v == "cpu" {
            return Ok(Self::Cpu);
        }
        if let Some(id) = v.strip_prefix("cuda:") {
            return Ok(Self::Cuda(id.parse()?));
        }
        if v == "cuda" {
            return Ok(Self::Cuda(0));
        }
        if let Some(id) = v.strip_prefix("metal:") {
            return Ok(Self::Metal(id.parse()?));
        }
        if v == "metal" {
            return Ok(Self::Metal(0));
        }
        anyhow::bail!("unsupported device: {value}")
    }
}

pub fn select_device(pref: DevicePreference) -> anyhow::Result<Device> {
    match pref {
        DevicePreference::Cpu => Ok(Device::Cpu),
        DevicePreference::Cuda(id) => cuda_device(id),
        DevicePreference::Metal(id) => metal_device(id),
        DevicePreference::Auto => auto_device(),
    }
}

pub fn auto_device() -> anyhow::Result<Device> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(dev) = Device::new_cuda(0) {
            return Ok(dev);
        }
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(dev) = Device::new_metal(0) {
            return Ok(dev);
        }
    }
    Ok(Device::Cpu)
}

pub fn cuda_device(id: usize) -> anyhow::Result<Device> {
    #[cfg(feature = "cuda")]
    {
        return Ok(Device::new_cuda(id)?);
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = id;
        anyhow::bail!("binary was built without cuda feature; rebuild with --no-default-features --features cuda")
    }
}

pub fn metal_device(id: usize) -> anyhow::Result<Device> {
    #[cfg(feature = "metal")]
    {
        return Ok(Device::new_metal(id)?);
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = id;
        anyhow::bail!("binary was built without metal feature; rebuild with --no-default-features --features metal")
    }
}

pub fn device_label(device: &Device) -> String {
    format!("{device:?}").to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_devices() {
        assert_eq!(
            DevicePreference::parse("auto").unwrap(),
            DevicePreference::Auto
        );
        assert_eq!(
            DevicePreference::parse("cuda:0").unwrap(),
            DevicePreference::Cuda(0)
        );
        assert_eq!(
            DevicePreference::parse("metal").unwrap(),
            DevicePreference::Metal(0)
        );
    }
}
